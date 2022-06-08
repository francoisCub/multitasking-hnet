from typing import Iterator, List

from pytorch_lightning import LightningDataModule
from torch import LongTensor, cat, randperm, stack, unique
from torch.utils.data import DataLoader, Sampler, SequentialSampler, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor)

from data.utils import get_sorted_dataset


class CifarBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool, num_tasks: int) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        n = len(sampler)
        if n % batch_size != 0:
            raise ValueError(
                "dataset size should be a multiple of the batch size")
        if n % num_tasks != 0:
            raise ValueError()
        init_idx = LongTensor(list(range(n))).split(n//num_tasks)
        new_idx = cat([idx_slice[randperm(len(idx_slice))] for idx_slice in init_idx])
        idx = new_idx.split(batch_size)
        base_idx = randperm(n//batch_size)
        self.idx = stack(idx)[base_idx]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.idx

    def __len__(self) -> int:
        return len(self.idx)


class LightningCifar(LightningDataModule):
    def __init__(self, batch_size, cifar=100,  data_dir=".data", num_class_per_task=10, n_classes=100, num_tasks=10, tasks=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = Compose([RandomCrop(32, padding=4, padding_mode='reflect'),
                                  RandomHorizontalFlip(),
                                  ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.test_transform = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.num_class_per_task = num_class_per_task
        self.n_classes = n_classes
        self.dataset_gen = CIFAR100 if cifar == 100 else CIFAR10
        if n_classes != num_tasks * num_class_per_task:
            raise ValueError()
        self.num_tasks = num_tasks
        self.tasks = tasks

    def prepare_data(self):
        self.dataset_gen(root=self.data_dir, train=True,
                 download=True, transform=ToTensor())
        self.dataset_gen(root=self.data_dir, train=False,
                 download=True, transform=ToTensor())

    def setup(self, stage):
        if stage == "fit" or stage is None:
            data_full = self.dataset_gen(root=self.data_dir, train=True,
                                 download=False, transform=self.transform)
            self.data_train, self.data_val = self.split_dataset(
                data_full)
            self.data_train = get_sorted_dataset(
                self.data_train, self.batch_size)
            self.data_val = get_sorted_dataset(self.data_val, self.batch_size)

        if stage == "test" or stage is None:
            self.data_test = self.dataset_gen(
                root=self.data_dir, train=False, download=False, transform=self.test_transform)
            self.data_test, _ = self.split_dataset(
                self.data_test, n_val=0)
            self.data_test = get_sorted_dataset(
                self.data_test, self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_sampler=self.batch_sampler(self.data_train), collate_fn=self.get_collate_fn())

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_sampler=self.batch_sampler(self.data_val), collate_fn=self.get_collate_fn())

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_sampler=self.batch_sampler(self.data_test), collate_fn=self.get_collate_fn())

    def batch_sampler(self, dataset):
        num_tasks = self.num_tasks if self.tasks is None else len(self.tasks)
        return CifarBatchSampler(SequentialSampler(dataset), batch_size=self.batch_size, drop_last=False, num_tasks=num_tasks)

    def get_collate_fn(self):
        num_class_per_task = self.num_class_per_task

        def collate_fn(batch):
            x = stack([x for x, _ in batch])
            y = LongTensor([y % num_class_per_task for _, y in batch])
            classes = LongTensor([y for _, y in batch])
            task = LongTensor([y//num_class_per_task for _, y in batch])
            task = unique(task)
            if len(task) > 1:
                raise RuntimeError(f"There should be one task per batch, got {task}")
            return x, y, classes, task
        return collate_fn

    def split_dataset(self, dataset, n_val=None):
        sorted_dataset = get_sorted_dataset(dataset, self.batch_size)
        if len(sorted_dataset) % self.n_classes != 0:
            raise ValueError()
        nbr_images_per_class = len(sorted_dataset) // self.n_classes
        # idx for a given class
        perm = randperm(len(sorted_dataset)//self.n_classes)
        if len(perm) % 10 != 0:
            raise ValueError()
        idx = perm[:len(perm)//10]
        if n_val is None:
            n_val = len(perm)//10
        val_idxs = []
        train_idxs = []
        if self.tasks is not None:
            classes = []
            for t in self.tasks:
                classes.extend([t*self.num_class_per_task+i for i in range(self.num_class_per_task)])
        else:
            classes = list(range(self.n_classes))
        for cls in classes:
            perm = randperm(nbr_images_per_class) + \
                nbr_images_per_class*cls  # idx for a given class
            if len(perm) % 10 != 0:
                raise ValueError()
            val_idx = perm[0:n_val].tolist()
            train_idx = perm[n_val:nbr_images_per_class].tolist()
            val_idxs.extend(val_idx)
            train_idxs.extend(train_idx)
        val_dataset = Subset(sorted_dataset, val_idxs)
        train_dataset = Subset(sorted_dataset, train_idxs)
        return train_dataset, val_dataset
