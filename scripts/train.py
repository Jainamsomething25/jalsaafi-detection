"""
Trains the oil-spill classifier and reports HONEST evaluation metrics.

Pipeline:
  1. Load train/val/test splits (built by prepare_splits.py).
  2. Compute class weights from the TRAIN split and use them in the loss
     function, so the model doesn't just learn to predict the majority
     class to minimize loss (a real risk with imbalanced data).
  3. Fine-tune a pretrained ResNet18 with early stopping on validation
     balanced accuracy (not raw accuracy - balanced accuracy treats both
     classes equally, so it isn't fooled by imbalance).
  4. Evaluate ONCE on the held-out test set (never touched during training)
     and report: accuracy, balanced accuracy, per-class precision/recall/F1,
     confusion matrix, ROC-AUC. Save plots.

Run with: venv\\Scripts\\python.exe scripts\\train.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.model_selection import train_test_split

from config import (
    SPLITS_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, EARLY_STOP_PATIENCE, MODEL_PATH,
    TRAINING_CURVE_PATH, CONFUSION_MATRIX_PATH, METRICS_REPORT_PATH,
    CLASS_NAMES, TRAIN_SUBSET_SIZE, RANDOM_SEED,
)
from scripts.dataset import SARDataset
from scripts.model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse-frequency class weights: rarer class gets a bigger weight in
    the loss, so misclassifying it costs more during training."""
    counts = np.bincount(labels)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, acc, bal_acc


def evaluate_test_set(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(oil)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_training_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_bal_acc"], label="train")
    axes[1].plot(epochs, history["val_bal_acc"], label="val")
    axes[1].set_title("Balanced Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.suptitle("Training curves - JalSaafi oil-spill detector")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    for split in ("train.csv", "val.csv", "test.csv"):
        if not (SPLITS_DIR / split).exists():
            print(
                f"ERROR: {SPLITS_DIR / split} not found. Run "
                f"scripts/prepare_splits.py first (it will also tell you if a "
                f"class is missing from the raw data)."
            )
            return

    device = get_device()
    print(f"Using device: {device}")

    train_ds = SARDataset(SPLITS_DIR / "train.csv", IMAGE_SIZE, train=True)
    val_ds = SARDataset(SPLITS_DIR / "val.csv", IMAGE_SIZE, train=False)
    test_ds = SARDataset(SPLITS_DIR / "test.csv", IMAGE_SIZE, train=False)

    # Optional stratified subsample of the TRAIN split only, to keep CPU
    # training time practical (val/test stay full-size for honest metrics).
    if TRAIN_SUBSET_SIZE is not None and TRAIN_SUBSET_SIZE < len(train_ds):
        full_labels = train_ds.labels()
        keep_idx, _ = train_test_split(
            range(len(train_ds.samples)),
            train_size=TRAIN_SUBSET_SIZE,
            stratify=full_labels,
            random_state=RANDOM_SEED,
        )
        train_ds.samples = [train_ds.samples[i] for i in keep_idx]
        new_counts = np.bincount([lbl for _, lbl in train_ds.samples])
        print(
            f"Subsampled train split to {len(train_ds)} images "
            f"(class counts: {dict(enumerate(new_counts))}) for faster CPU training."
        )

    if len(set(train_ds.labels())) < 2:
        print("ERROR: training split has fewer than 2 classes. Cannot train a binary classifier yet.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_weights = compute_class_weights(train_ds.labels()).to(device)
    print(f"Class weights (no_oil, oil): {class_weights.tolist()}")

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "val_loss": [], "train_bal_acc": [], "val_bal_acc": []}
    best_val_bal_acc = -1.0
    epochs_without_improvement = 0

    print(f"\nTraining for up to {NUM_EPOCHS} epochs (early stop patience={EARLY_STOP_PATIENCE})...\n")
    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_bal_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_bal_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_bal_acc"].append(train_bal_acc)
        history["val_bal_acc"].append(val_bal_acc)

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} bal_acc {train_bal_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} bal_acc {val_bal_acc:.3f}"
        )

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> new best val balanced accuracy, saved model to {MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no val improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed / 60:.1f} minutes.")

    plot_training_curves(history, TRAINING_CURVE_PATH)
    print(f"Saved training curves to {TRAINING_CURVE_PATH}")

    # --- Final evaluation on the untouched test set, using the BEST checkpoint ---
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    y_true, y_pred, y_prob = evaluate_test_set(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present in y_true

    report_lines = [
        "=== JalSaafi Oil-Spill Detector - Test Set Evaluation ===",
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


if __name__ == "__main__":
    main()
