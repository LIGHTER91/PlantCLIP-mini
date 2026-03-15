import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T

from model import PlantCLIP


MODEL_PATH = "plantclip_model.pt"
CSV_PATH = "metadata.csv"
INDEX_EMB_PATH = "image_embeddings.pt"
INDEX_META_PATH = "image_index.csv"
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    model = PlantCLIP(
        embed_dim=checkpoint["embed_dim"],
        text_model_name=checkpoint["text_model_name"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_image_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def encode_image(model, image_path, image_transform, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    image_embedding = model.image_encoder(image_tensor)
    return image_embedding.squeeze(0).cpu()


def main():
    print("Using device:", DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV introuvable: {CSV_PATH}")

    model = load_model(MODEL_PATH, DEVICE)
    image_transform = get_image_transform()
    df = pd.read_csv(CSV_PATH)

    embeddings = []
    valid_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building image index"):
        image_path = row["image_path"]

        if not os.path.exists(image_path):
            continue

        try:
            emb = encode_image(model, image_path, image_transform, DEVICE)
            embeddings.append(emb)
            valid_rows.append(row.to_dict())
        except Exception as e:
            print(f"Erreur sur {image_path}: {e}")

    if len(embeddings) == 0:
        raise RuntimeError("Aucun embedding image n'a pu être calculé.")

    image_embeddings = torch.stack(embeddings, dim=0)
    valid_df = pd.DataFrame(valid_rows)

    torch.save(image_embeddings, INDEX_EMB_PATH)
    valid_df.to_csv(INDEX_META_PATH, index=False)

    print(f"Embeddings sauvegardés dans: {INDEX_EMB_PATH}")
    print(f"Index metadata sauvegardé dans: {INDEX_META_PATH}")
    print(f"Nombre d'images indexées: {len(valid_df)}")


if __name__ == "__main__":
    main()