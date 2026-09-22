"""
Final test-set evaluation for a checkpoint already saved by train.py.

Used here because training was manually stopped after epoch 8 (user was
short on time) — the best checkpoint on disk is from epoch 7 (val balanced
accuracy 0.969, the last epoch that improved before training was cut off).
This script evaluates that checkpoint on the held-out test set (never seen
during training) and produces the same honest metrics report + plots that
train.py would have produced at the end of a full run.

Run with: venv\\Scripts\\python.exe scripts\\evaluate.py
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SPLITS_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, MODEL_PATH, CLASS_NAMES
from scripts.dataset import SARDataset
from scripts.model import load_model_for_inference
from scripts.train import (
    evaluate_test_set, plot_training_curves, plot_confusion_matrix,
)
from config import TRAINING_CURVE_PATH, CONFUSION_MATRIX_PATH, METRICS_REPORT_PATH
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)

# Epoch-by-epoch history transcribed from the training run's console output
# (epochs 1-8, run manually stopped after epoch 8 by user request; best
# checkpoint is from epoch 7). Not re-derived from a log file because the
# run was interrupted before it could write one.
HISTORY = {
    "train_loss":   [0.3898, 0.2416, 0.1567, 0.1366, 0.1362, 0.1018, 0.1015, 0.1253],
    "val_loss":     [0.2000, 0.1564, 0.1239, 0.1209, 0.1661, 0.0951, 0.0709, 0.1323],
    "train_bal_acc":[0.815,  0.908,  0.940,  0.950,  0.951,  0.963,  0.963,  0.957],
    "val_bal_acc":  [0.927,  0.946,  0.957,  0.953,  0.948,  0.959,  0.969,  0.956],
}
BEST_EPOCH = 7  # 1-indexed; this is the checkpoint actually saved at MODEL_PATH


def main():
    device = torch.device("cpu")

    if not MODEL_PATH.exists():
        print(f"ERROR: no checkpoint found at {MODEL_PATH}.")
        return

    test_ds = SARDataset(SPLITS_DIR / "test.csv", IMAGE_SIZE, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Loading checkpoint (best model, epoch {BEST_EPOCH}) from {MODEL_PATH} ...")
    model = load_model_for_inference(MODEL_PATH, device=device)

    print(f"Evaluating on held-out test set ({len(test_ds)} images, never used in training)...")
    y_true, y_pred, y_prob = evaluate_test_set(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    report_lines = [
        "=== JalSaafi Oil-Spill Detector - Test Set Evaluation ===",
        f"Checkpoint: epoch {BEST_EPOCH} (best val balanced accuracy during training: 0.969)",
        "NOTE: training was manually stopped after epoch 8/15 (user time constraint), "
        "using early-stop patience 5 with a max of 15 epochs. Epoch 7 was the last "
        "epoch that improved validation balanced accuracy, so it is the saved checkpoint.",
        f"Test set size: {len(y_true)}",
        f"Overall accuracy: {acc:.3f}",
        f"Balanced accuracy: {bal_acc:.3f}",
        f"ROC-AUC: {auc:.3f}",
        "",
        "Per-class precision/recall/F1:",
        report,
        "Confusion matrix (rows=actual, cols=predicted):",
        f"  {CLASS_NAMES}",
        str(cm),
        "",
        "NOTE: this is a proof-of-concept classifier trained on a limited "
        "dataset. SAR oil-spill detection is known to be confounded by "
        "'look-alikes' (low wind zones, algae films, wave shadows) that "
        "mimic oil's dark radar signature. Metrics above reflect performance "
        "on this dataset only and should not be read as real-world accuracy.",
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    METRICS_REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nSaved metrics report to {METRICS_REPORT_PATH}")

    plot_confusion_matrix(cm, CLASS_NAMES, CONFUSION_MATRIX_PATH)
    print(f"Saved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

    plot_training_curves(HISTORY, TRAINING_CURVE_PATH)
    print(f"Saved training curves plot to {TRAINING_CURVE_PATH}")


if __name__ == "__main__":
    main()
