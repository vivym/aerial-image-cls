from pathlib import Path
from typing import Optional, Tuple

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms as T
from torchvision.transforms.functional import InterpolationMode

from .transforms import RandomMixup, RandomCutmix
from .presets import ClassificationPresetTrain, ClassificationPresetEval


class AerialImageDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 16,
        num_classes: int = 1000,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        train_crop_size: int = 224,
        val_crop_size: int = 224,
        val_resize_size: int = 256,
        interpolation: str = "bilinear",
        auto_augment_policy: Optional[str] = None,
        random_erase_prob: float = 0.0,
        ra_magnitude: int = 9,
        augmix_severity: int = 3,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_cache: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="num_classes")

        self.root_path = Path(root_path)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.train_crop_size = train_crop_size
        self.val_crop_size = val_crop_size
        self.val_resize_size = val_resize_size
        self.interpolation = InterpolationMode(interpolation)
        self.auto_augment_policy = auto_augment_policy
        self.random_erase_prob = random_erase_prob
        self.ra_magnitude = ra_magnitude
        self.augmix_severity = augmix_severity
        self.mean = mean
        self.std = std
        self.use_cache = use_cache

    def train_dataloader(self):
        collate_fn = None
        mixup_transforms = []

        mixup_transforms = []
        if self.mixup_alpha > 0.0:
            mixup_transforms.append(
                RandomMixup(num_classes=self.num_classes, p=1.0, alpha=self.mixup_alpha)
            )
        if self.cutmix_alpha > 0.0:
            mixup_transforms.append(
                RandomCutmix(num_classes=self.num_classes, p=1.0, alpha=self.cutmix_alpha)
            )
        if mixup_transforms:
            mixupcutmix = T.RandomChoice(mixup_transforms)

            def collate_fn(batch):
                return mixupcutmix(*default_collate(batch))

        data_path = self.root_path / "train"
        dataset = datasets.ImageFolder(
            data_path,
            transform=ClassificationPresetTrain(
                crop_size=self.train_crop_size,
                mean=self.mean,
                std=self.std,
                interpolation=self.interpolation,
                auto_augment_policy=self.auto_augment_policy,
                random_erase_prob=self.random_erase_prob,
                ra_magnitude=self.ra_magnitude,
                augmix_severity=self.augmix_severity,
            )
        )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        data_path = self.root_path / "val"
        dataset = datasets.ImageFolder(
            self.root_path / "val",
            transform=ClassificationPresetEval(
                crop_size=self.val_crop_size,
                resize_size=self.val_resize_size,
                mean=self.mean,
                std=self.std,
                interpolation=self.interpolation,
            )
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
