
"""Create reproducible duplicate-aware train/validation/test manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(raw_dir: Path):
    classes = {}
    hash_to_class = {}
    duplicate_count = 0

    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        groups = defaultdict(list)

        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(path) as image:
                    image.verify()
                digest = image_hash(path)
            except (OSError, ValueError):
                continue

            # Same exact image must not have conflicting labels.
            previous_class = hash_to_class.get(digest)
            if previous_class is not None and previous_class != class_dir.name:
                raise ValueError(
                    f"Same image found in different classes: "
                    f"{previous_class} and {class_dir.name}\n{path}"
                )

            hash_to_class[digest] = class_dir.name
            groups[digest].append(str(path.resolve()))

        if groups:
            classes[class_dir.name] = list(groups.values())
            duplicate_count += sum(len(paths) - 1 for paths in groups.values())

    if not classes:
        raise FileNotFoundError(f"No valid class images found in {raw_dir}")

    print("Exact duplicate copies removed from manifests:", duplicate_count)
    return classes


def build_manifests(
    raw_dir: Path,
    output_dir: Path,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:

    if validation_size < 0 or test_size < 0:
        raise ValueError("Split sizes must be non-negative")

    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")

    classes = collect_images(raw_dir)
    rng = random.Random(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = {"train": [], "val": [], "test": []}
    class_names = sorted(classes)

    for label, class_name in enumerate(class_names):
        groups = classes[class_name][:]
        rng.shuffle(groups)

        n = len(groups)
        test_count = max(1, round(n * test_size))
        val_count = max(1, round(n * validation_size))

        if test_count + val_count >= n:
            test_count = 1
            val_count = 0

        test_groups = groups[:test_count]
        val_groups = groups[test_count:test_count + val_count]
        train_groups = groups[test_count + val_count:]

        split_groups = {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups,
        }

        for split, selected_groups in split_groups.items():
            for group in selected_groups:
                # One representative path per exact duplicate group.
                rows[split].append((group[0], label, class_name))

    for split, split_rows in rows.items():
        with (output_dir / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(("path", "label", "class_name"))
            writer.writerows(split_rows)

    (output_dir / "classes.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    counts = {split: len(items) for split, items in rows.items()}
    print("New manifest counts:", counts)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print(build_manifests(**vars(args)))


if __name__ == "__main__":
    main()
