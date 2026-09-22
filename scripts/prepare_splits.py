"""
Builds a stratified train/val/test split from the raw class folders and
writes it out as CSVs (data/splits/{train,val,test}.csv).

Stratified = each split preserves the overall class ratio, so a small class
doesn't accidentally end up entirely in one split. We split test out first,
then split train/val from what's left, so all three splits are stratified
relative to the full dataset.

Splits are written to disk (not regenerated on every training run) so that:
  1. The test set stays identical across experiments (fair comparison).
  2. You can inspect exactly which images ended up where.
"""

import csv
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, CLASS_MAP, SPLITS_DIR, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED


def collect_samples():
    samples = []  # list of (filepath_str, label)
    for class_folder, label in CLASS_MAP.items():
        folder = RAW_DATA_DIR / class_folder
        if not folder.exists():
            print(f"WARNING: {folder} does not exist, skipping this class.")
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                samples.append((str(p), label))
    return samples


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(rows)


def main():
    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    samples = collect_samples()
    labels_present = sorted(set(label for _, label in samples))

    print(f"Collected {len(samples)} labeled images across classes: {labels_present}")

    if len(labels_present) < 2:
        print(
            "\nERROR: fewer than 2 classes found in the raw data. Cannot build a "
            "stratified train/val/test split for a binary classifier - need both "
            "'no oil' and 'oil' images present on disk first.\n"
            "Re-run this script once Class_1 (oil) images have been added."
        )
        return

    filepaths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # First carve out the test set.
    train_val_fp, test_fp, train_val_lb, test_lb = train_test_split(
        filepaths, labels,
        test_size=TEST_FRAC,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Then split what's left into train/val.
    val_relative_frac = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_fp, val_fp, train_lb, val_lb = train_test_split(
        train_val_fp, train_val_lb,
        test_size=val_relative_frac,
        stratify=train_val_lb,
        random_state=RANDOM_SEED,
    )

    write_csv(SPLITS_DIR / "train.csv", zip(train_fp, train_lb))
    write_csv(SPLITS_DIR / "val.csv", zip(val_fp, val_lb))
    write_csv(SPLITS_DIR / "test.csv", zip(test_fp, test_lb))

    def summarize(name, lbls):
        counts = {c: lbls.count(c) for c in sorted(set(lbls))}
        print(f"  {name}: {len(lbls)} images, class counts = {counts}")

    print("\nSplit sizes:")
    summarize("train", train_lb)
    summarize("val", val_lb)
    summarize("test", test_lb)
    print(f"\nWrote split CSVs to: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
