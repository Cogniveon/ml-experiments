import torch
import torchvision
import lightning.pytorch as pl
import torchvision.transforms.v2 as transforms

from torch.utils.data import DataLoader


class Flowers102DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 8, num_workers: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize((224, 224)),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

    def prepare_data(self):
        torchvision.datasets.Flowers102(
            root=self.data_dir, split="train", download=True
        )
        torchvision.datasets.Flowers102(root=self.data_dir, split="test", download=True)
        torchvision.datasets.Flowers102(root=self.data_dir, split="val", download=True)

    def setup(self, stage: str):
        if stage == "fit":
            self.flowers_train = torchvision.datasets.Flowers102(
                root=self.data_dir,
                split="train",
                download=True,
                transform=self.transform,
            )

        if stage == "predict" or stage == "test":
            self.flowers_test = torchvision.datasets.Flowers102(
                root=self.data_dir,
                split="test",
                download=True,
                transform=self.transform,
            )

        if stage == "fit":
            self.flowers_val = torchvision.datasets.Flowers102(
                root=self.data_dir, split="val", download=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.flowers_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.flowers_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.flowers_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.flowers_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
