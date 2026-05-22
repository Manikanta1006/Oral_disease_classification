import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import (
    DISEASE_DISPLAY_NAMES,
    DISEASE_NAMES,
    IMAGE_SIZE,
    REGION_DISPLAY_NAMES,
    REGION_NAMES,
)


MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "oral_region_disease_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

_model = None
_labels = None


def display_name(label: str, names: dict[str, str]):
    return names.get(label, label.replace("_", " ").title())


def load_artifacts():
    global _model, _labels

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train first with: python backend/train.py"
            )
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _labels is None:
        if not LABELS_PATH.exists():
            raise FileNotFoundError(
                f"Labels not found at {LABELS_PATH}. Train first with: python backend/train.py"
            )

        _labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        expected = {
            "regions": REGION_NAMES,
            "diseases": DISEASE_NAMES,
        }
        actual = {
            "regions": _labels.get("regions"),
            "diseases": _labels.get("diseases"),
        }
        if actual != expected:
            raise ValueError(
                "Saved label order does not match this app. "
                f"Expected {expected}, found {actual}. Retrain the model."
            )

    return _model, _labels


def preprocess_image(image_path: str | Path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def unpack_predictions(model: tf.keras.Model, predictions):
    if isinstance(predictions, dict):
        return predictions["region"][0], predictions["disease"][0]

    if isinstance(predictions, (list, tuple)):
        prediction_map = {
            output_name: prediction
            for output_name, prediction in zip(model.output_names, predictions)
        }
        return prediction_map["region"][0], prediction_map["disease"][0]

    raise ValueError("Model must return region and disease predictions.")


def probability_map(labels: list[str], scores):
    return {label: float(score) for label, score in zip(labels, scores)}


def predict_image(image_path: str | Path):
    model, labels = load_artifacts()
    batch = preprocess_image(image_path)
    predictions = model.predict(batch, verbose=0)
    region_scores, disease_scores = unpack_predictions(model, predictions)

    region_index = int(np.argmax(region_scores))
    disease_index = int(np.argmax(disease_scores))

    region_label = labels["regions"][region_index]
    disease_label = labels["diseases"][disease_index]
    region_confidence = float(region_scores[region_index])
    disease_confidence = float(disease_scores[disease_index])

    detected_region = display_name(region_label, REGION_DISPLAY_NAMES)
    possible_disease = display_name(disease_label, DISEASE_DISPLAY_NAMES)

    return {
        "detected_region": detected_region,
        "possible_disease": possible_disease,
        "prediction_value": disease_confidence,
        "prediction_percent": f"{disease_confidence * 100:.1f}%",
        "region": {
            "label": region_label,
            "name": detected_region,
            "confidence": region_confidence,
        },
        "disease": {
            "label": disease_label,
            "name": possible_disease,
            "confidence": disease_confidence,
        },
        "probabilities": {
            "regions": probability_map(labels["regions"], region_scores),
            "diseases": probability_map(labels["diseases"], disease_scores),
        },
        # Kept for older UI code or quick scripts that still read class/confidence.
        "class": disease_label,
        "confidence": disease_confidence,
    }


def print_text_result(result: dict):
    print(f"Detected Region: {result['detected_region']}")
    print(f"Possible Disease: {result['possible_disease']}")
    print(f"Prediction Value: {result['prediction_percent']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict oral region and disease from an image."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the full JSON result.")
    args = parser.parse_args()

    output = predict_image(args.image)
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print_text_result(output)
