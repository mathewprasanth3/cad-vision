"""
src/models/resnet_branch.py

Single view feature extractor using pretrained ResNet18.
Takes one image [3, 224, 224] and outputs a 512-dim feature vector.

This is used as a building block inside MVCNN — one branch per view.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetBranch(nn.Module):

    def __init__(self, pretrained: bool = True, frozen: bool = False):
        super().__init__()

        # load pretrained resnet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # remove the final classification layer
        # resnet18 originally: conv layers -> avgpool -> fc(512 -> 1000)
        # we want:             conv layers -> avgpool -> 512-dim vector
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # optionally freeze pretrained weights
        # useful at the start of training to avoid destroying pretrained features
        if frozen:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: [batch, 3, 224, 224]
        x = self.features(x)
        # after features: [batch, 512, 1, 1]
        x = x.flatten(start_dim=1)
        # after flatten: [batch, 512]
        return x


if __name__ == "__main__":
    branch = ResNetBranch(pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    out = branch(dummy)
    print(f"input shape:  {dummy.shape}")
    print(f"output shape: {out.shape}")
    print("resnet branch working correctly" if out.shape == (4, 512) else "something wrong")