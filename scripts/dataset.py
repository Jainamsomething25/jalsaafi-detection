"""
PyTorch Dataset for the SAR oil-spill images.

Channel handling: Sentinel-1 SAR is physically a SINGLE backscatter channel
(unlike a camera photo). But we're using transfer learning from ResNet18,
which was pretrained on ImageNet (3-channel RGB photos) - its early
convolution filters and learned feature hierarchy are only useful to us if
we feed it 3-channel input in the shape it expects.

So every image, regardless of how it's stored on disk (this dataset's JPEGs
happen to already be grayscale values replicated into R/G/B), is explicitly
converted to single-channel grayscale and then replicated into 3 identical
channels. That gives us a deterministic, format-independent pipeline and
lets us reuse the pretrained weights untouched, rather than surgically
resizing the first conv layer to 1 channel (which would throw away most of
its pretrained weights for that layer).
"""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# ImageNet normalization stats - required because we're using ImageNet
# pretrained weights; the network expects inputs normalized this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Training gets augmentation (flips/rotation) to fight overfitting and
    help the minority class generalize; val/test do not (we want to measure
    performance on unaltered images)."""
    ops = [transforms.Grayscale(num_output_channels=3)]

    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ]

    ops += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class SARDataset(Dataset):
    """Reads a split CSV (columns: filepath,label) produced by
    scripts/prepare_splits.py."""

    def __init__(self, csv_path: Path, image_size: int, train: bool):
        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filepath"], int(row["label"])))
        self.transform = build_transforms(image_size, train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")  # normalize source mode before Grayscale()
            tensor = self.transform(img)
        return tensor, label

    def labels(self):
        return [label for _, label in self.samples]
