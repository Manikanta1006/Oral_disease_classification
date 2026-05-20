import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

from config import CLASS_NAMES, IMAGE_EXTENSIONS, IMAGE_SIZE


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "oral_disease_model.h5"
LABELS_PATH = MODEL_DIR / "class_names.json"

BATCH_SIZE = 32
SEED = 42


def count_images(class_dir: Path):
    return sum(
        1
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_split(split_dir: Path, split_name: str):
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing dataset/{split_name} folder.")

    missing = [name for name in CLASS_NAMES if not (split_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing class folders in dataset/{split_name}: {', '.join(missing)}"
        )

    counts = {name: count_images(split_dir / name) for name in CLASS_NAMES}
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(
            f"No images found in dataset/{split_name} for: {', '.join(empty)}. "
            "Put JPG, JPEG, PNG, or WEBP files inside each class folder."
        )

    return counts


def build_datasets(data_dir: Path, batch_size: int):
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"

    train_counts = validate_split(train_dir, "train")
    valid_counts = validate_split(valid_dir, "valid")

    print("Training image counts:", train_counts)
    print("Validation image counts:", valid_counts)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        seed=SEED,
        label_mode="categorical",
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        seed=SEED,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    valid_ds = valid_ds.cache().prefetch(autotune)
    return train_ds, valid_ds, class_names


def build_model(num_classes: int, learning_rate: float):
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

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="oral_disease_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, valid_ds, class_names = build_datasets(args.dataset_dir, args.batch_size)
    model = build_model(len(class_names), args.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
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
    LABELS_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved class names to: {LABELS_PATH}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train oral disease classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
