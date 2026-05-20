import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import CLASS_NAMES, IMAGE_SIZE


MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "oral_disease_model.h5"
LABELS_PATH = MODEL_DIR / "class_names.json"

_model = None
_class_names = None


def load_artifacts():
    global _model, _class_names

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train first with: python backend/train.py"
            )
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _class_names is None:
        if not LABELS_PATH.exists():
            raise FileNotFoundError(
                f"Class labels not found at {LABELS_PATH}. Train first with: python backend/train.py"
            )

        _class_names = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        if _class_names != CLASS_NAMES:
            raise ValueError(
                "Saved class label order does not match this app. "
                f"Expected {CLASS_NAMES}, found {_class_names}. Retrain the model."
            )

    return _model, _class_names


def preprocess_image(image_path: str | Path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def predict_image(image_path: str | Path):
    model, class_names = load_artifacts()
    batch = preprocess_image(image_path)
    predictions = model.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(predictions))

    return {
        "class": class_names[best_index],
        "confidence": float(predictions[best_index]),
        "probabilities": {
            class_names[index]: float(score)
            for index, score in enumerate(predictions)
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict oral disease from an image.")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    result = predict_image(args.image)
    print(json.dumps(result, indent=2))
