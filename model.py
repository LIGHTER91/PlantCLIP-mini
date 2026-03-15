import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", embed_dim: int = 256):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.projection = nn.Linear(hidden_size, embed_dim)

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


class PlantCLIP(nn.Module):
    def __init__(self, embed_dim: int = 256, text_model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(model_name=text_model_name, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        scale = self.logit_scale.exp()
        logits_per_image = scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text, image_embeds, text_embeds


def clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.size(0)
    targets = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) / 2