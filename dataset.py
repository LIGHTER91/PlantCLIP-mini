import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PlantCLIPDataset(Dataset):
    def __init__(self, csv_path, tokenizer, image_transform=None, max_length=32):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        text = row["text"]
        label = row["label"]

        if self.image_transform:
            image = self.image_transform(image)

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": text_tokens["input_ids"].squeeze(0),
            "attention_mask": text_tokens["attention_mask"].squeeze(0),
            "text": text,
            "label": label
        }