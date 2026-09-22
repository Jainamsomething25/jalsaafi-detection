"""
Inspects the raw SAR dataset BEFORE any training happens.

Why this exists: training on a dataset you haven't looked at is how you end
up with silent bugs (corrupt files, mismatched image modes, or a class that
doesn't exist yet). This script reports, per class folder found under
RAW_DATA_DIR:
  - image count
  - file format(s)
  - dimensions found (and whether they're consistent)
  - color mode: true single-channel grayscale vs. RGB-with-identical-channels
    vs. genuine RGB (SAR is physically single-band, so this matters for how
    we build the model's input layer)
  - corrupt / unreadable files
  - class imbalance ratio

Run with: venv\\Scripts\\python.exe scripts\\inspect_dataset.py
"""

import sys
from collections import Counter
from pathlib import Path

from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, CLASS_MAP, OUTPUTS_DIR


def is_grayscale_stored_as_rgb(img: Image.Image, sample_pixels: int = 2000) -> bool:
    """Check whether an RGB image actually has R==G==B everywhere (i.e. it's
    grayscale data that was saved in a 3-channel format), by sampling pixels
    rather than checking the whole image (faster, and JPEG noise means exact
    equality isn't required - we check "very close")."""
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    if flat.shape[0] > sample_pixels:
        idx = np.random.choice(flat.shape[0], sample_pixels, replace=False)
        flat = flat[idx]
    diffs = flat.max(axis=1).astype(int) - flat.min(axis=1).astype(int)
    return float(np.mean(diffs <= 3)) > 0.98  # allow small JPEG compression noise


def inspect_class_folder(folder: Path):
    exts = Counter()
    dims = Counter()
    modes = Counter()
    grayscale_as_rgb = 0
    genuine_rgb = 0
    corrupt = []
    file_sizes_kb = []

    files = sorted([p for p in folder.iterdir() if p.is_file()])
    for p in files:
        exts[p.suffix.lower()] += 1
        file_sizes_kb.append(p.stat().st_size / 1024)
        try:
            with Image.open(p) as img:
                img.verify()  # cheap corruption check
            with Image.open(p) as img:  # reopen: verify() leaves file unusable
                dims[img.size] += 1
                modes[img.mode] += 1
                if img.mode == "RGB":
                    if is_grayscale_stored_as_rgb(img):
                        grayscale_as_rgb += 1
                    else:
                        genuine_rgb += 1
        except Exception as e:
            corrupt.append((p.name, str(e)))

    return {
        "folder": folder.name,
        "count": len(files),
        "extensions": exts,
        "dimensions": dims,
        "modes": modes,
        "grayscale_as_rgb": grayscale_as_rgb,
        "genuine_rgb": genuine_rgb,
        "corrupt": corrupt,
        "file_sizes_kb": file_sizes_kb,
    }


def main():
    print(f"Scanning: {RAW_DATA_DIR}\n")

    if not RAW_DATA_DIR.exists():
        print(f"ERROR: RAW_DATA_DIR does not exist: {RAW_DATA_DIR}")
        return

    all_subdirs = {p.name: p for p in RAW_DATA_DIR.iterdir() if p.is_dir()}
    # Only treat folders named in CLASS_MAP as classes. RAW_DATA_DIR also
    # contains a "sample" folder (a small preview subset, duplicates of the
    # first few chips) which is NOT part of the labeled training data and
    # must not be counted as a third class.
    found_class_dirs = {name: path for name, path in all_subdirs.items() if name in CLASS_MAP}
    other_dirs = [name for name in all_subdirs if name not in CLASS_MAP]

    report_lines = []
    report_lines.append(f"Dataset root: {RAW_DATA_DIR}")
    report_lines.append(f"Class folders expected: {list(CLASS_MAP.keys())}")
    report_lines.append(f"Class folders found:    {list(found_class_dirs.keys())}")
    if other_dirs:
        report_lines.append(f"Other (non-class) folders present, ignored: {other_dirs}")
    report_lines.append("")

    missing = [c for c in CLASS_MAP if c not in found_class_dirs]
    if missing:
        report_lines.append(
            f"** WARNING: missing class folder(s): {missing}. **\n"
            "This dataset currently has only ONE class present. A binary "
            "classifier CANNOT be trained until both classes exist - training "
            "will be skipped/blocked until this is resolved.\n"
        )

    results = []
    for name, path in found_class_dirs.items():
        print(f"Inspecting {name} ...")
        r = inspect_class_folder(path)
        results.append(r)

    total_images = sum(r["count"] for r in results)

    for r in results:
        report_lines.append(f"--- {r['folder']} ---")
        report_lines.append(f"  Images: {r['count']}")
        report_lines.append(f"  Extensions: {dict(r['extensions'])}")
        report_lines.append(f"  Dimensions found: {dict(r['dimensions'])}")
        report_lines.append(f"  PIL modes found: {dict(r['modes'])}")
        report_lines.append(
            f"  Grayscale-stored-as-RGB (R~=G~=B): {r['grayscale_as_rgb']}"
        )
        report_lines.append(f"  Genuine multi-channel RGB: {r['genuine_rgb']}")
        if r["file_sizes_kb"]:
            sizes = np.array(r["file_sizes_kb"])
            report_lines.append(
                f"  File size (KB): min={sizes.min():.1f} "
                f"mean={sizes.mean():.1f} max={sizes.max():.1f}"
            )
        if r["corrupt"]:
            report_lines.append(f"  CORRUPT/UNREADABLE FILES: {len(r['corrupt'])}")
            for fname, err in r["corrupt"][:10]:
                report_lines.append(f"    - {fname}: {err}")
        else:
            report_lines.append("  Corrupt files: none found")
        report_lines.append("")

    report_lines.append("=== Class balance ===")
    for r in results:
        pct = 100 * r["count"] / total_images if total_images else 0
        report_lines.append(f"  {r['folder']}: {r['count']} images ({pct:.1f}%)")
    if len(results) == 2 and all(r["count"] > 0 for r in results):
        counts = sorted([r["count"] for r in results])
        imbalance_ratio = counts[1] / counts[0]
        report_lines.append(f"  Imbalance ratio (majority/minority): {imbalance_ratio:.2f}x")
        if imbalance_ratio > 1.5:
            report_lines.append(
                "  -> Meaningful class imbalance detected. Training pipeline "
                "must use weighted loss / weighted sampling (it does)."
            )

    report_text = "\n".join(str(l) for l in report_lines)
    print("\n" + report_text)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "dataset_inspection_report.txt"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
