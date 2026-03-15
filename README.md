
# PlantCLIP-mini

Implémentation simplifiée d'un modèle **vision–langage de type CLIP** appliqué à la **détection de maladies des plantes** à partir d'images de feuilles et de descriptions textuelles.

Le projet apprend un **espace d'embedding multimodal partagé** entre images et texte afin de permettre :
- la correspondance image ↔ texte
- la recherche d’images à partir d’une requête en langage naturel
- l’exploration sémantique d’images agricoles

---

# Aperçu

Les modèles vision–langage modernes (CLIP, BLIP, etc.) apprennent à aligner des images et du texte dans un espace vectoriel commun.

Dans ce projet :

- **Encodeur image** : ResNet18
- **Encodeur texte** : DistilBERT
- **Dimension des embeddings** : 256
- **Apprentissage** : contrastif (CLIP loss)

Le modèle est entraîné sur le dataset **PlantVillage** avec des descriptions textuelles générées automatiquement à partir des labels.

---

# Dataset

Dataset utilisé : **PlantVillage**

Caractéristiques :

- ~54 000 images de feuilles
- 38 classes (plante + maladie)
- 14 espèces de plantes
- images RGB

Exemples de classes :

```
Tomato___Early_blight
Apple___healthy
Grape___Black_rot
```

Les labels sont transformés en descriptions textuelles afin d'entraîner le modèle vision-langage.

Exemple de caption générée :

```
a close-up photo of a tomato leaf showing early blight symptoms
```

---

# Architecture du modèle

## Encodeur image

Backbone :

```
ResNet18
```

Projection linéaire vers un embedding de 256 dimensions.

## Encodeur texte

Backbone :

```
DistilBERT
```

Pooling moyen des tokens puis projection vers 256 dimensions.

Les deux embeddings sont normalisés avant l'entraînement.

---

# Apprentissage contrastif

Le modèle est entraîné avec une **loss contrastive de type CLIP**.

Objectif :

- rapprocher les embeddings image–texte correspondants
- éloigner les autres combinaisons dans le batch

Formule simplifiée :

L = CE(image→texte) + CE(texte→image)

Chaque batch agit également comme ensemble de **négatifs implicites**.

---

# Configuration d'entraînement

Paramètres principaux :

```
Batch size : 32
Learning rate : 1e-4
Optimizer : AdamW
Scheduler : CosineAnnealingLR
Epochs : 10
Embedding dimension : 256
```

Le scheduler **CosineAnnealingLR** diminue progressivement le learning rate pour stabiliser la convergence.

---

# Résultats d'entraînement

Résultats observés pendant l'entraînement :

| Epoch | Train Loss | Val Loss | Image→Texte Acc | Texte→Image Acc |
|------|-----------|----------|----------------|----------------|
| 1 | 0.8455 | 0.7555 | 0.5748 | 0.5796 |
| 2 | 0.7422 | 0.7343 | 0.5823 | 0.5825 |
| 3 | 0.7279 | 0.7271 | 0.5795 | 0.5821 |
| 4 | 0.7204 | 0.7265 | 0.5819 | 0.5846 |
| 5 | 0.7184 | 0.7189 | 0.5778 | 0.5848 |
| 6 | 0.7154 | 0.7173 | 0.5823 | 0.5816 |
| 7 | 0.7055 | 0.7197 | 0.5807 | 0.5800 |

Meilleure validation observée :

```
Validation loss ≈ 0.717
Image → Texte accuracy ≈ 0.58
Texte → Image accuracy ≈ 0.58
```

---

# Structure du projet

```
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
```

---

# Pipeline d'utilisation

## 1. Générer les descriptions

```
python create_metadata.py
```

Cela crée le fichier :

```
metadata.csv
```

---

## 2. Entraîner le modèle

```
python train_clip.py
```

Le meilleur modèle est sauvegardé dans :

```
plantclip_model.pt
```

---

## 3. Construire l'index d'embeddings image

```
python build_index.py
```

Cela génère :

```
image_embeddings.pt
image_index.csv
```

---

## 4. Rechercher des images avec une requête texte

Exemple :

```
python inference.py --query "tomato leaf with early blight"
```

Le système retourne les images les plus proches dans l’espace d'embedding.

---

# Limitations

- captions générées automatiquement
- dataset limité en diversité textuelle
- encodeur image relativement simple
- évaluation retrieval encore basique

---

# Références

- Radford et al. – Learning Transferable Visual Models From Natural Language Supervision (CLIP)
- PlantVillage dataset
- PyTorch
- HuggingFace Transformers

---

# Licence

MIT License
