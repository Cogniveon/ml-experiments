import torch
import torchvision
import lightning.pytorch as pl
import torchvision.transforms.v2 as transforms

from torch.utils.data import DataLoader


class CIFAR10DataModule(pl.LightningDataModule):
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
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            self.cifar_train = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform,
            )

        if stage == "predict" or stage == "test":
            self.cifar_test = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transform,
            )

        if stage == "fit":
            self.cifar_val = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
