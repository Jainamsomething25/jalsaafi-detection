"""
Inference module for the JalSaafi oil-spill detector.

Importable API (for plugging into a web dashboard later):
    from predict import predict_image, locate_oil_region

    result = predict_image("some_sar_chip.jpg")
    # -> {"oil_detected": True, "confidence": 0.87, "label": "oil", ...}

    if result["oil_detected"]:
        loc = locate_oil_region("some_sar_chip.jpg")
        # -> {"centroid_xy": (123, 87), "bbox": (x_min, y_min, x_max, y_max), ...}

Can also be run directly from the command line:
    venv\\Scripts\\python.exe predict.py path\\to\\image.jpg
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES
from scripts.dataset import build_transforms
from scripts.model import load_model_for_inference

_model = None  # lazily loaded once, reused across calls
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run scripts/train.py first."
            )
        _model = load_model_for_inference(MODEL_PATH, device=_device)
    return _model


def predict_image(image_path: str) -> dict:
    """Classifies a single SAR image chip as oil / no-oil.

    Returns a dict with:
        oil_detected: bool
        label: "oil" or "no_oil"
        confidence: float in [0, 1] - the model's probability for the
                    PREDICTED class (not necessarily P(oil))
        prob_oil: float in [0, 1] - raw P(oil), useful if a caller wants to
                  apply their own decision threshold instead of argmax
    """
    model = _get_model()
    transform = build_transforms(IMAGE_SIZE, train=False)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        tensor = transform(img).unsqueeze(0).to(_device)  # add batch dim

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "oil_detected": pred_idx == 1,
        "label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "prob_oil": float(probs[1]),
    }


def _otsu_threshold(gray: np.ndarray) -> float:
    """Simple Otsu's method implemented from scratch with numpy: finds the
    pixel-intensity threshold that best separates the image into two groups
    (dark region vs. background) by maximizing between-class variance.
    No extra dependencies (opencv/scipy) needed for this."""
    hist, bin_edges = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()

    sum_total = np.dot(np.arange(256), hist)
    sum_bg, weight_bg = 0.0, 0.0
    best_thresh, best_variance = 0, -1.0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_variance > best_variance:
            best_variance = between_variance
            best_thresh = t

    return float(best_thresh)


def locate_oil_region(image_path: str) -> dict:
    """APPROXIMATE oil-region localization for a SAR chip already flagged
    as containing oil. Oil slicks dampen surface roughness, so they show up
    as DARK (low-backscatter) patches in SAR imagery. We exploit that with
    simple global thresholding (Otsu) + centroid-of-dark-pixels - this is
    NOT a trained segmentation model, just a heuristic, and should be
    labeled as approximate anywhere it's shown (e.g. a dashboard pin).

    Coordinates are in the ORIGINAL image's pixel space (not the resized
    224x224 model input), and also returned normalized to [0,1] so they can
    be mapped onto real-world coordinates by the caller regardless of chip
    resolution.
    """
    with Image.open(image_path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.uint8)

    h, w = gray.shape
    threshold = _otsu_threshold(gray)
    dark_mask = gray < threshold  # oil = dark patches

    if not dark_mask.any():
        return {
            "found": False,
            "note": "No sufficiently dark region found - approximate "
                    "thresholding method could not localize a spill.",
        }

    ys, xs = np.nonzero(dark_mask)
    centroid_x, centroid_y = float(xs.mean()), float(ys.mean())
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    dark_fraction = float(dark_mask.mean())

    return {
        "found": True,
        "centroid_xy": (centroid_x, centroid_y),
        "centroid_xy_normalized": (centroid_x / w, centroid_y / h),
        "bbox_xyxy": bbox,
        "image_size": (w, h),
        "dark_area_fraction": dark_fraction,
        "method": "otsu_threshold_centroid",
        "note": "APPROXIMATE - derived from simple intensity thresholding, "
                "not a trained segmentation model. Treat as a rough pin, "
                "not a precise spill boundary.",
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    result = predict_image(image_path)
    print(f"Prediction: {result['label']} (oil_detected={result['oil_detected']})")
    print(f"Confidence: {result['confidence']:.1%}  (P(oil)={result['prob_oil']:.1%})")

    if result["oil_detected"]:
        loc = locate_oil_region(image_path)
        if loc["found"]:
            print(f"Approx. spill centroid (pixels): {loc['centroid_xy']}")
            print(f"Approx. bounding box (x_min,y_min,x_max,y_max): {loc['bbox_xyxy']}")
            print(f"Note: {loc['note']}")
        else:
            print(loc["note"])
