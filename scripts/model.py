"""
Model definition: ResNet18 pretrained on ImageNet, transfer-learned for
binary oil / no-oil classification.

Why ResNet18: it's small enough to train quickly on a CPU or modest GPU,
and its ImageNet-pretrained low/mid-level features (edges, textures,
gradients) transfer reasonably well to SAR imagery even though SAR looks
very different from natural photos - the alternative (training a CNN from
scratch on ~1-2k images) would badly overfit.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, freeze_backbone: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        # Optional: freeze everything except the new classifier head.
        # Left off by default because our dataset, while small, benefits
        # from fine-tuning the later conv blocks too (SAR textures differ
        # a lot from ImageNet photos).
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer: ResNet18's default head
    # predicts 1000 ImageNet classes; we only need 2 (no_oil, oil).
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def load_model_for_inference(checkpoint_path, device="cpu") -> nn.Module:
    model = build_model(num_classes=2)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
