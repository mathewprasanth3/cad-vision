"""
src/models/mvcnn.py

Multi-View CNN for machining feature recognition.
Takes 4 rendered views of a CAD part and predicts the machining feature class.

Architecture:
    4 views → 4x ResNet18 branches → max pooling → FC layers → class prediction
"""

import torch
import torch.nn as nn
from src.models.resnet_branch import ResNetBranch


class MVCNN(nn.Module):

    def __init__(
        self,
        num_classes: int = 24,
        pretrained: bool = True,
        frozen_branches: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes

        # one resnet branch per view
        self.branch = ResNetBranch(pretrained=pretrained, frozen=frozen_branches)

        # classifier head — takes fused 512-dim vector and predicts class
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [batch, 4, 3, 224, 224]
        batch_size, num_views, c, h, w = x.shape

        # reshape to process all views through the branch at once
        x = x.view(batch_size * num_views, c, h, w)
        # x: [batch*4, 3, 224, 224]

        # extract features from each view
        features = self.branch(x)
        # features: [batch*4, 512]

        # reshape back to separate views
        features = features.view(batch_size, num_views, 512)
        # features: [batch, 4, 512]

        # max pooling across views
        # take the strongest feature from all 4 views for each dimension
        features, _ = torch.max(features, dim=1)
        # features: [batch, 512]

        # classify
        out = self.classifier(features)
        # out: [batch, num_classes]

        return out


if __name__ == "__main__":
    model = MVCNN(num_classes=24, pretrained=False)

    dummy = torch.randn(2, 4, 3, 224, 224)
    out = model(dummy)

    print(f"input shape:  {dummy.shape}")
    print(f"output shape: {out.shape}")
    print(f"expected:     torch.Size([2, 24])")
    print("mvcnn working correctly" if out.shape == (2, 24) else "something wrong")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\ntotal parameters:     {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")