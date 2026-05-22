REGION_NAMES = ["inner_cheek", "gums", "tongue", "teeth", "lips", "palate"]
DISEASE_NAMES = ["gingivitis", "healthy", "oral_ulcer", "tooth_decay"]

REGION_DISPLAY_NAMES = {
    "inner_cheek": "Inner cheek",
    "gums": "Gums",
    "tongue": "Tongue",
    "teeth": "Teeth",
    "lips": "Lips",
    "palate": "Palate",
}

DISEASE_DISPLAY_NAMES = {
    "gingivitis": "Gingivitis",
    "healthy": "Healthy",
    "oral_ulcer": "Oral ulcer",
    "tooth_decay": "Tooth decay",
}

REGION_FOLDER_ALIASES = {
    "inner_cheek": ["inner cheek", "inner-cheek"],
}

DISEASE_FOLDER_ALIASES = {
    "oral_ulcer": ["mouth_ulcer", "oral ulcer", "mouth ulcer"],
}

IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
