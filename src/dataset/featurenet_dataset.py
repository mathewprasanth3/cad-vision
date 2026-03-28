"""
src/dataset/featurenet_dataset.py

PyTorch dataset for the FeatureNet machining feature dataset.
Loads 4-view rendered PNG images for each CAD model.

Each sample returns:
    images: tensor of shape [4, 3, 224, 224]  (4 views)
    label:  int class index (0-23)
"""

import os
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


VIEWS = ["front", "side", "top", "isometric"]

CLASS_NAMES = [
    "0_chamfer",
    "1_through_hole",
    "2_triangular_passage",
    "3_rectangular_passage",
    "4_circular_through_slot",
    "5_triangular_through_slot",
    "6_rectangular_through_slot",
    "7_rectangular_blind_slot",
    "8_triangular_pocket",
    "9_rectangular_pocket",
    "10_circular_end_pocket",
    "11_triangular_blind_step",
    "12_circular_blind_step",
    "13_rectangular_blind_step",
    "14_rectangular_through_step",
    "15_two_sides_through_step",
    "16_slanted_through_step",
    "17_round",
    "18_v_circular_end_blind_slot",
    "19_h_circular_end_blind_slot",
    "20_rectangular_passage_2",
    "21_b_passage",
    "22_pocket_2",
    "23_o_ring",
]


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class FeatureNetDataset(Dataset):

    def __init__(
        self,
        renders_dir: str | Path,
        split: Literal["train", "val", "test"] = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.renders_dir = Path(renders_dir)
        self.split = split
        self.transform = get_transforms(split)

        self.samples = self._load_samples(train_ratio, val_ratio, seed)

    def _load_samples(self, train_ratio, val_ratio, seed):
        all_samples = []

        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = self.renders_dir / class_name
            if not class_dir.exists():
                # try to find folder by prefix number
                matches = [
                    d for d in self.renders_dir.iterdir()
                    if d.is_dir() and d.name.startswith(f"{label}_")
                ]
                if not matches:
                    continue
                class_dir = matches[0]

            # find all unique model stems in this class
            front_files = sorted(class_dir.glob("*_front.png"))
            for front_file in front_files:
                stem = front_file.stem.replace("_front", "")
                all_samples.append((class_dir, stem, label))

        # reproducible split
        import random
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        n = len(all_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if self.split == "train":
            return all_samples[:n_train]
        elif self.split == "val":
            return all_samples[n_train:n_train + n_val]
        else:
            return all_samples[n_train + n_val:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_dir, stem, label = self.samples[idx]

        images = []
        for view in VIEWS:
            img_path = class_dir / f"{stem}_{view}.png"
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            images.append(img)

        images = torch.stack(images)  # [4, 3, 224, 224]

        return images, label


def get_dataloaders(
    renders_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
):
    train_dataset = FeatureNetDataset(renders_dir, split="train", seed=seed)
    val_dataset = FeatureNetDataset(renders_dir, split="val", seed=seed)
    test_dataset = FeatureNetDataset(renders_dir, split="test", seed=seed)

    print(f"train: {len(train_dataset)} samples")
    print(f"val:   {len(val_dataset)} samples")
    print(f"test:  {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader