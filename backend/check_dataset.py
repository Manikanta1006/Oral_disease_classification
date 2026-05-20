from pathlib import Path

from config import CLASS_NAMES, IMAGE_EXTENSIONS


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
SPLITS = ["train", "valid", "test"]


def count_images(class_dir: Path):
    return sum(
        1
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    has_error = False

    for split in SPLITS:
        split_dir = DATASET_DIR / split
        print(f"\n{split.upper()}")

        if not split_dir.is_dir():
            print(f"  Missing folder: {split_dir}")
            has_error = True
            continue

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                print(f"  {class_name}: missing folder")
                has_error = True
                continue

            count = count_images(class_dir)
            print(f"  {class_name}: {count} images")
            if split in {"train", "valid"} and count == 0:
                has_error = True

    if has_error:
        raise SystemExit(
            "\nDataset check failed. Add images to every train/valid class folder "
            "before training."
        )

    print("\nDataset check passed.")


if __name__ == "__main__":
    main()
