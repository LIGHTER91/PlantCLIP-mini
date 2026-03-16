# PlantCLIP-mini

A simplified implementation of a **CLIP-style vision--language model**
applied to **plant disease detection** using leaf images and textual
descriptions.

The project learns a **shared multimodal embedding space** between
images and text, enabling:

-   image ↔ text matching
-   image retrieval using natural language queries
-   semantic exploration of agricultural images

------------------------------------------------------------------------

# Overview

Modern vision--language models (such as CLIP or BLIP) learn to align
images and text within a shared vector space.

In this project:

-   **Image encoder**: ResNet18
-   **Text encoder**: DistilBERT
-   **Embedding dimension**: 256
-   **Training objective**: contrastive learning (CLIP loss)

The model is trained on the **PlantVillage dataset**, where textual
descriptions are automatically generated from class labels.

------------------------------------------------------------------------

# Dataset

Dataset used: **PlantVillage**

Characteristics:

-   \~54,000 leaf images
-   38 classes (plant + disease)
-   14 plant species
-   RGB images

Example classes:

    Tomato___Early_blight
    Apple___healthy
    Grape___Black_rot

Labels are converted into textual descriptions to train the
vision--language model.

Example generated caption:

    a close-up photo of a tomato leaf showing early blight symptoms

------------------------------------------------------------------------

# Model Architecture

## Image Encoder

Backbone:

    ResNet18

Followed by a linear projection into a **256-dimensional embedding
space**.

## Text Encoder

Backbone:

    DistilBERT

Mean pooling is applied to the token embeddings, followed by a
projection into the **256-dimensional embedding space**.

Both embeddings are **L2-normalized** before training.

------------------------------------------------------------------------

# Contrastive Training

The model is trained using a **CLIP-style contrastive loss**.

Objective:

-   bring matching **image--text embeddings closer**
-   push **non-matching pairs apart**

Simplified loss formulation:

    L = CE(image→text) + CE(text→image)

Each batch also acts as a set of **implicit negatives**.

------------------------------------------------------------------------

# Training Configuration

Main parameters:

    Batch size : 32
    Learning rate : 1e-4
    Optimizer : AdamW
    Scheduler : CosineAnnealingLR
    Epochs : 10
    Embedding dimension : 256

The **CosineAnnealingLR scheduler** progressively reduces the learning
rate to stabilize convergence.

------------------------------------------------------------------------

# Training Results

Observed training results:

  Epoch   Train Loss   Val Loss   Image→Text Acc   Text→Image Acc
  ------- ------------ ---------- ---------------- ----------------
  1       0.8455       0.7555     0.5748           0.5796
  2       0.7422       0.7343     0.5823           0.5825
  3       0.7279       0.7271     0.5795           0.5821
  4       0.7204       0.7265     0.5819           0.5846
  5       0.7184       0.7189     0.5778           0.5848
  6       0.7154       0.7173     0.5823           0.5816
  7       0.7055       0.7197     0.5807           0.5800

Best validation performance:

    Validation loss ≈ 0.717
    Image → Text accuracy ≈ 0.58
    Text → Image accuracy ≈ 0.58

------------------------------------------------------------------------

# Project Structure

    plantclip-mini
    │
    ├── data/
    │   └── images/
    │
    ├── create_metadata.py
    ├── dataset.py
    ├── model.py
    ├── train_clip.py
    ├── build_index.py
    ├── inference.py
    │
    ├── metadata.csv
    ├── image_embeddings.pt
    ├── image_index.csv
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# Usage Pipeline

## 1. Generate captions

    python create_metadata.py

This creates the file:

    metadata.csv

------------------------------------------------------------------------

## 2. Train the model

    python train_clip.py

The best model is saved as:

    plantclip_model.pt

------------------------------------------------------------------------

## 3. Build the image embedding index

    python build_index.py

This generates:

    image_embeddings.pt
    image_index.csv

------------------------------------------------------------------------

## 4. Search images using a text query

Example:

    python inference.py --query "tomato leaf with early blight"

The system returns the closest images in the embedding space.

------------------------------------------------------------------------

# Limitations

-   captions are automatically generated
-   limited textual diversity in the dataset
-   relatively simple image encoder
-   basic retrieval evaluation

------------------------------------------------------------------------

# References

-   Radford et al. --- Learning Transferable Visual Models From Natural
    Language Supervision (CLIP)
-   PlantVillage Dataset
-   PyTorch
-   HuggingFace Transformers

------------------------------------------------------------------------

# License

MIT License
