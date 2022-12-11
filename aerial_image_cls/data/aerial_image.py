from pathlib import Path
from typing import Optional, Tuple, List, Callable

import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as T


class AerialImageDatasetImpl(Dataset):
    def __init__(
        self,
        root_path: Path,
        labels: List[int],
        repeat: int = 1,
        transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.labels = labels
        self.transform = transform

        self.paths = list((root_path / "images").glob("*.tif")) * repeat

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        label_path = image_path.parent.parent / "label" / image_path.name

        img = Image.open(image_path)

        if self.transform is not None:
            img = self.transform(img)

        label = Image.open(label_path)
        label = np.asarray(label).astype(np.int64)

        remapped_label = np.zeros_like(label)
        for i, label_i in enumerate(self.labels):
            remapped_label[label == label_i] = i

        return img, remapped_label, image_path.name


class AerialImageDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        labels: List[int],
        repeat: int = 1,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 16,
        mean: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5),
        std: Tuple[float, float, float, float] = (0.225, 0.225, 0.225, 0.225),
    ):
        super().__init__()
        self.save_hyperparameters(ignore="num_classes")

        self.root_path = Path(root_path)
        self.labels = labels
        self.repeat = repeat
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std

    def train_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

        dataset = AerialImageDatasetImpl(
            self.root_path / "train",
            labels=self.labels,
            repeat=self.repeat,
            transform=transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
        dataset = AerialImageDatasetImpl(
            self.root_path / "val",
            labels=self.labels,
            repeat=self.repeat,
            transform=transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    image = Image.open("data/250mqgc/images/000000002814.tif").convert("RGB")

    label = Image.open("data/250mqgc/label/000000002814.tif")

    print(np.asarray(image).shape)
    print(np.unique(np.asarray(label)))


if __name__ == "__main__":
    main()
