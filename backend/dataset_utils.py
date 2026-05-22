from dataclasses import dataclass
from pathlib import Path

from config import (
    DISEASE_FOLDER_ALIASES,
    DISEASE_NAMES,
    IMAGE_EXTENSIONS,
    REGION_FOLDER_ALIASES,
    REGION_NAMES,
)


@dataclass(frozen=True)
class LabeledImage:
    path: Path
    region_index: int
    disease_index: int
    region_name: str
    disease_name: str


def label_folder_candidates(label: str, aliases: dict[str, list[str]]):
    return [label, *aliases.get(label, [])]


def image_files(folder: Path):
    if not folder.is_dir():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_split_examples(split_dir: Path):
    examples: list[LabeledImage] = []

    for region_index, region_name in enumerate(REGION_NAMES):
        for region_folder_name in label_folder_candidates(
            region_name, REGION_FOLDER_ALIASES
        ):
            region_dir = split_dir / region_folder_name
            if not region_dir.is_dir():
                continue

            for disease_index, disease_name in enumerate(DISEASE_NAMES):
                for disease_folder_name in label_folder_candidates(
                    disease_name, DISEASE_FOLDER_ALIASES
                ):
                    disease_dir = region_dir / disease_folder_name
                    for image_path in image_files(disease_dir):
                        examples.append(
                            LabeledImage(
                                path=image_path,
                                region_index=region_index,
                                disease_index=disease_index,
                                region_name=region_name,
                                disease_name=disease_name,
                            )
                        )

    return examples


def summarize_examples(examples: list[LabeledImage]):
    region_counts = {name: 0 for name in REGION_NAMES}
    disease_counts = {name: 0 for name in DISEASE_NAMES}

    for example in examples:
        region_counts[example.region_name] += 1
        disease_counts[example.disease_name] += 1

    return {
        "total": len(examples),
        "regions": region_counts,
        "diseases": disease_counts,
    }


def validate_split(data_dir: Path, split_name: str, require_all_labels: bool = True):
    split_dir = data_dir / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset/{split_name} folder.")

    examples = collect_split_examples(split_dir)
    summary = summarize_examples(examples)

    if summary["total"] == 0:
        raise ValueError(
            f"No images found in dataset/{split_name}. Expected folders like "
            f"dataset/{split_name}/inner_cheek/oral_ulcer/image.jpg"
        )

    if require_all_labels:
        empty_regions = [
            name for name, count in summary["regions"].items() if count == 0
        ]
        empty_diseases = [
            name for name, count in summary["diseases"].items() if count == 0
        ]

        errors = []
        if empty_regions:
            errors.append(f"regions with no images: {', '.join(empty_regions)}")
        if empty_diseases:
            errors.append(f"diseases with no images: {', '.join(empty_diseases)}")

        if errors:
            raise ValueError(f"dataset/{split_name} is incomplete: {'; '.join(errors)}")

    return examples, summary
