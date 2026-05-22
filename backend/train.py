import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

from config import DISEASE_NAMES, IMAGE_SIZE, REGION_NAMES
from dataset_utils import LabeledImage, validate_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "oral_region_disease_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

BATCH_SIZE = 32
SEED = 42


def print_split_summary(split_name: str, summary: dict):
    print(f"\n{split_name.upper()} images: {summary['total']}")
    print("  Regions:")
    for name, count in summary["regions"].items():
        print(f"    {name}: {count}")

    print("  Diseases:")
    for name, count in summary["diseases"].items():
        print(f"    {name}: {count}")


def read_image_with_pil(path: tf.Tensor):
    image_path = path.numpy().decode("utf-8")
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32)


def load_and_resize_image(path: tf.Tensor, labels: dict[str, tf.Tensor]):
    image = tf.py_function(read_image_with_pil, [path], tf.float32)
    image.set_shape((*IMAGE_SIZE, 3))
    return image, labels


def make_dataset(
    examples: list[LabeledImage],
    batch_size: int,
    is_training: bool,
):
    paths = [str(example.path) for example in examples]
    region_labels = np.array(
        [example.region_index for example in examples], dtype=np.int32
    )
    disease_labels = np.array(
        [example.disease_index for example in examples], dtype=np.int32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            {
                "region": region_labels,
                "disease": disease_labels,
            },
        )
    )
    dataset = dataset.map(load_and_resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(len(examples), 1000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_datasets(data_dir: Path, batch_size: int):
    train_examples, train_summary = validate_split(data_dir, "train")
    valid_examples, valid_summary = validate_split(data_dir, "valid")

    print_split_summary("train", train_summary)
    print_split_summary("valid", valid_summary)

    train_ds = make_dataset(train_examples, batch_size, is_training=True)
    valid_ds = make_dataset(valid_examples, batch_size, is_training=False)

    return train_ds, valid_ds


def build_model(
    num_regions: int,
    num_diseases: int,
    learning_rate: float,
):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    region_output = layers.Dense(
        num_regions,
        activation="softmax",
        name="region",
    )(x)
    disease_output = layers.Dense(
        num_diseases,
        activation="softmax",
        name="disease",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "region": region_output,
            "disease": disease_output,
        },
        name="oral_region_disease_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "region": "sparse_categorical_crossentropy",
            "disease": "sparse_categorical_crossentropy",
        },
        metrics={
            "region": ["accuracy"],
            "disease": ["accuracy"],
        },
    )
    return model


def train(args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, valid_ds = build_datasets(args.dataset_dir, args.batch_size)
    model = build_model(
        num_regions=len(REGION_NAMES),
        num_diseases=len(DISEASE_NAMES),
        learning_rate=args.learning_rate,
    )

    monitor_mode = "max" if "accuracy" in args.monitor_metric else "min"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor=args.monitor_metric,
            mode=monitor_mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    LABELS_PATH.write_text(
        json.dumps(
            {
                "regions": REGION_NAMES,
                "diseases": DISEASE_NAMES,
                "image_size": list(IMAGE_SIZE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved labels to: {LABELS_PATH}")

    final_region_accuracy = history.history.get("val_region_accuracy", [None])[-1]
    final_disease_accuracy = history.history.get("val_disease_accuracy", [None])[-1]
    if final_region_accuracy is not None:
        print(f"Final validation region accuracy: {final_region_accuracy:.4f}")
    if final_disease_accuracy is not None:
        print(f"Final validation disease accuracy: {final_disease_accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN for oral region and disease prediction."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--monitor-metric", default="val_disease_accuracy")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
