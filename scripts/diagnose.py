import sys
import time
from pathlib import Path

def log(msg):
    print(f"[{time.time():.1f}] {msg}", flush=True)

log("start")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
log(f"torch imported, version={torch.__version__}")

from torchvision import models
log("torchvision.models imported")

weights = models.ResNet18_Weights.IMAGENET1K_V1
log("weights enum obtained")

model = models.resnet18(weights=weights)
log("resnet18 model constructed with pretrained weights")

import torch.nn as nn
model.fc = nn.Linear(model.fc.in_features, 2)
log("fc layer replaced")

from config import SPLITS_DIR, IMAGE_SIZE
from scripts.dataset import SARDataset
log("dataset module imported")

ds = SARDataset(SPLITS_DIR / "train.csv", IMAGE_SIZE, train=True)
log(f"dataset object created, len={len(ds)}")

log("loading first sample...")
x, y = ds[0]
log(f"first sample loaded, shape={x.shape}, label={y}")

log("loading 8 samples...")
for i in range(8):
    x, y = ds[i]
log("8 samples loaded")

from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
log("dataloader created")

it = iter(loader)
log("got iterator, fetching first batch...")
images, labels = next(it)
log(f"first batch fetched: {images.shape}")

log("running forward pass...")
model.eval()
with torch.no_grad():
    out = model(images)
log(f"forward pass done: {out.shape}")

log("ALL DONE")
