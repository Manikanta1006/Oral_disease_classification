from pathlib import Path

from config import DISEASE_NAMES, REGION_NAMES
from dataset_utils import collect_split_examples, summarize_examples, validate_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
SPLITS = ["train", "valid", "test"]


def print_summary(split: str, summary: dict):
    print(f"\n{split.upper()}")
    print(f"  total: {summary['total']} images")

    print("  regions:")
    for name in REGION_NAMES:
        print(f"    {name}: {summary['regions'][name]}")

    print("  diseases:")
    for name in DISEASE_NAMES:
        print(f"    {name}: {summary['diseases'][name]}")


def main():
    has_error = False

    for split in SPLITS:
        split_dir = DATASET_DIR / split
        if split in {"train", "valid"}:
            try:
                _examples, summary = validate_split(DATASET_DIR, split)
            except (FileNotFoundError, ValueError) as exc:
                print(f"\n{split.upper()}")
                print(f"  {exc}")
                has_error = True
                continue
        else:
            examples = collect_split_examples(split_dir)
            summary = summarize_examples(examples)

        print_summary(split, summary)

    if has_error:
        raise SystemExit(
            "\nDataset check failed. Use folders like "
            "dataset/train/inner_cheek/oral_ulcer/image.jpg for train and valid."
        )

    print("\nDataset check passed.")


if __name__ == "__main__":
    main()
