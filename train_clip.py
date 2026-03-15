import os
import csv
import random
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from dataset import PlantCLIPDataset
from model import PlantCLIP, clip_loss


if not torch.cuda.is_available():
    raise RuntimeError("CUDA non détecté. Ce script doit tourner uniquement sur GPU.")

torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0")

CSV_PATH = "metadata.csv"
MODEL_SAVE_PATH = "plantclip_model.pt"
METRICS_CSV_PATH = "training_metrics.csv"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 224
EMBED_DIM = 256
TEXT_MODEL_NAME = "distilbert-base-uncased"
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    correct_i2t = 0
    correct_t2i = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        logits_per_image, logits_per_text, _, _ = model(images, input_ids, attention_mask)
        loss = clip_loss(logits_per_image, logits_per_text)

        total_loss += loss.item()
        total_batches += 1

        targets = torch.arange(images.size(0), device=device)
        pred_i2t = logits_per_image.argmax(dim=1)
        pred_t2i = logits_per_text.argmax(dim=1)

        correct_i2t += (pred_i2t == targets).sum().item()
        correct_t2i += (pred_t2i == targets).sum().item()
        total_samples += images.size(0)

    return {
        "loss": total_loss / max(total_batches, 1),
        "image_to_text_acc": correct_i2t / max(total_samples, 1),
        "text_to_image_acc": correct_t2i / max(total_samples, 1),
    }


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits_per_image, logits_per_text, _, _ = model(images, input_ids, attention_mask)
        loss = clip_loss(logits_per_image, logits_per_text)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def main():
    set_seed(SEED)

    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Using device:", DEVICE)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} introuvable.")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    image_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    dataset = PlantCLIPDataset(
        csv_path=CSV_PATH,
        tokenizer=tokenizer,
        image_transform=image_transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = PlantCLIP(
        embed_dim=EMBED_DIM,
        text_model_name=TEXT_MODEL_NAME
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
    best_val_loss = float("inf")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    with open(METRICS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "i2t_acc", "t2i_acc"])

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
            val_metrics = evaluate(model, val_loader, DEVICE)

            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss:   {val_metrics['loss']:.4f}")
            print(f"I->T acc:   {val_metrics['image_to_text_acc']:.4f}")
            print(f"T->I acc:   {val_metrics['text_to_image_acc']:.4f}")
            scheduler.step()
            print(f"Current LR: {scheduler.get_last_lr()[0]:.8f}")
            writer.writerow([
                epoch + 1,
                f"{train_loss:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['image_to_text_acc']:.6f}",
                f"{val_metrics['text_to_image_acc']:.6f}",
            ])
            f.flush()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "embed_dim": EMBED_DIM,
                        "text_model_name": TEXT_MODEL_NAME
                    },
                    MODEL_SAVE_PATH
                )
                print(f"Best model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()