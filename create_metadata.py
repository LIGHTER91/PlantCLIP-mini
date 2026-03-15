import os
import random
import pandas as pd

DATA_DIR = "data/images"
OUTPUT_CSV = "metadata.csv"
SEED = 42

random.seed(SEED)

# Descriptions plus naturelles par maladie
DISEASE_TEXT = {
    "healthy": [
        "healthy",
        "without visible disease symptoms",
        "with normal leaf appearance",
        "showing no visible sign of infection"
    ],
    "Apple_scab": [
        "showing apple scab symptoms",
        "with dark scab lesions typical of apple scab",
        "affected by apple scab disease"
    ],
    "Black_rot": [
        "showing black rot symptoms",
        "with dark lesions typical of black rot",
        "affected by black rot disease"
    ],
    "Cedar_apple_rust": [
        "showing cedar apple rust symptoms",
        "with orange rust-like lesions typical of cedar apple rust",
        "affected by cedar apple rust disease"
    ],
    "Powdery_mildew": [
        "showing powdery mildew symptoms",
        "with powdery fungal growth typical of powdery mildew",
        "affected by powdery mildew disease"
    ],
    "Cercospora_leaf_spot Gray_leaf_spot": [
        "showing cercospora leaf spot symptoms",
        "with gray leaf spot lesions typical of cercospora infection",
        "affected by gray leaf spot disease"
    ],
    "Common_rust": [
        "showing common rust symptoms",
        "with rust-colored lesions typical of common rust",
        "affected by common rust disease"
    ],
    "Northern_Leaf_Blight": [
        "showing northern leaf blight symptoms",
        "with elongated lesions typical of northern leaf blight",
        "affected by northern leaf blight disease"
    ],
    "Esca_(Black_Measles)": [
        "showing esca symptoms",
        "with black measles symptoms typical of esca disease",
        "affected by esca disease"
    ],
    "Leaf_blight_(Isariopsis_Leaf_Spot)": [
        "showing leaf blight symptoms",
        "with lesions typical of isariopsis leaf spot",
        "affected by leaf blight disease"
    ],
    "Haunglongbing_(Citrus_greening)": [
        "showing citrus greening symptoms",
        "affected by huanglongbing disease",
        "with symptoms typical of citrus greening"
    ],
    "Bacterial_spot": [
        "showing bacterial spot symptoms",
        "with bacterial spot lesions",
        "affected by bacterial spot disease"
    ],
    "Early_blight": [
        "showing early blight symptoms",
        "with brown necrotic lesions typical of early blight",
        "affected by early blight disease"
    ],
    "Late_blight": [
        "showing late blight symptoms",
        "with dark infected areas typical of late blight",
        "affected by late blight disease"
    ],
    "Leaf_Mold": [
        "showing leaf mold symptoms",
        "with signs of leaf mold infection",
        "affected by leaf mold disease"
    ],
    "Septoria_leaf_spot": [
        "showing septoria leaf spot symptoms",
        "with necrotic spotting typical of septoria leaf spot",
        "affected by septoria leaf spot disease"
    ],
    "Spider_mites Two-spotted_spider_mite": [
        "showing spider mite damage",
        "affected by two-spotted spider mites",
        "with visible symptoms of spider mite infestation"
    ],
    "Target_Spot": [
        "showing target spot symptoms",
        "with lesions typical of target spot disease",
        "affected by target spot disease"
    ],
    "Tomato_Yellow_Leaf_Curl_Virus": [
        "showing yellow leaf curl virus symptoms",
        "affected by tomato yellow leaf curl virus",
        "with symptoms typical of tomato yellow leaf curl virus infection"
    ],
    "Tomato_mosaic_virus": [
        "showing mosaic virus symptoms",
        "affected by tomato mosaic virus",
        "with symptoms typical of tomato mosaic virus infection"
    ]
}

# Templates visuels
VIEW_TEMPLATES = [
    "a close-up photo of a {plant} leaf {desc}",
    "an rgb image of a {plant} leaf {desc}",
    "a plant pathology image of a {plant} leaf {desc}",
    "a crop leaf image showing a {plant} leaf {desc}",
    "a detailed image of a {plant} leaf {desc}"
]

# Fallback si la maladie n’est pas explicitement décrite
GENERIC_DISEASE_TEMPLATES = [
    "showing {disease} symptoms",
    "affected by {disease}",
    "with visible signs of {disease}",
    "with symptoms consistent with {disease}"
]


def clean_plant_name(name: str) -> str:
    return name.replace("_", " ").lower()


def clean_disease_name(name: str) -> str:
    return name.replace("_", " ").replace("(", "").replace(")", "").lower()


def get_description(disease_raw: str) -> str:
    if disease_raw in DISEASE_TEXT:
        return random.choice(DISEASE_TEXT[disease_raw])

    disease_clean = clean_disease_name(disease_raw)
    return random.choice(GENERIC_DISEASE_TEMPLATES).format(disease=disease_clean)


def build_caption(plant_raw: str, disease_raw: str) -> str:
    plant_clean = clean_plant_name(plant_raw)
    desc = get_description(disease_raw)
    template = random.choice(VIEW_TEMPLATES)
    return template.format(plant=plant_clean, desc=desc)


def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dossier introuvable: {DATA_DIR}")

    rows = []

    for label in sorted(os.listdir(DATA_DIR)):
        label_path = os.path.join(DATA_DIR, label)

        if not os.path.isdir(label_path):
            continue

        if "___" not in label:
            print(f"Label ignoré car format inattendu: {label}")
            continue

        plant_raw, disease_raw = label.split("___", 1)

        plant_clean = clean_plant_name(plant_raw)
        disease_clean = clean_disease_name(disease_raw)

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            if not os.path.isfile(img_path):
                continue

            text = build_caption(plant_raw, disease_raw)

            rows.append({
                "image_path": img_path.replace("\\", "/"),
                "label": label,
                "plant": plant_clean,
                "disease": disease_clean,
                "text": text
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"{OUTPUT_CSV} créé avec {len(df)} images")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()