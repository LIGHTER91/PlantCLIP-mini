import os
import argparse
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer

from model import PlantCLIP


MODEL_PATH = "plantclip_model.pt"
INDEX_EMB_PATH = "image_embeddings.pt"
INDEX_META_PATH = "image_index.csv"
TOP_K = 5

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

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["text_model_name"])
    return model, tokenizer


@torch.no_grad()
def encode_text(model, tokenizer, text, device):
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    text_embedding = model.text_encoder(input_ids, attention_mask)
    return text_embedding.squeeze(0).cpu()


@torch.no_grad()
def retrieve_top_k(model, tokenizer, query_text, image_matrix, metadata_df, top_k, device):
    text_emb = encode_text(model, tokenizer, query_text, device)
    similarities = image_matrix @ text_emb

    top_k = min(top_k, len(metadata_df))
    scores, indices = torch.topk(similarities, k=top_k)

    results = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        row = metadata_df.iloc[idx]
        results.append({
            "image_path": row["image_path"],
            "label": row["label"],
            "text": row["text"],
            "score": score
        })
    return results


def show_results(query_text, results):
    n = len(results)
    plt.figure(figsize=(4 * n, 4.8))

    for i, item in enumerate(results, start=1):
        image = Image.open(item["image_path"]).convert("RGB")
        plt.subplot(1, n, i)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"{item['label']}\nscore={item['score']:.3f}", fontsize=9)

    plt.suptitle(f"Query: {query_text}", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help='Exemple: "tomato leaf with early blight"')
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    print("Using device:", DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
    if not os.path.exists(INDEX_EMB_PATH):
        raise FileNotFoundError(f"Index embeddings introuvable: {INDEX_EMB_PATH}")
    if not os.path.exists(INDEX_META_PATH):
        raise FileNotFoundError(f"Index metadata introuvable: {INDEX_META_PATH}")

    model, tokenizer = load_model(MODEL_PATH, DEVICE)
    image_matrix = torch.load(INDEX_EMB_PATH, map_location="cpu")
    metadata_df = pd.read_csv(INDEX_META_PATH)

    results = retrieve_top_k(
        model=model,
        tokenizer=tokenizer,
        query_text=args.query,
        image_matrix=image_matrix,
        metadata_df=metadata_df,
        top_k=args.top_k,
        device=DEVICE
    )

    print("\nTop results:")
    for i, item in enumerate(results, start=1):
        print(f"{i}. score={item['score']:.4f} | label={item['label']} | path={item['image_path']}")

    if not args.no_plot:
        show_results(args.query, results)


if __name__ == "__main__":
    main()