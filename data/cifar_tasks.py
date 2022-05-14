from collections import defaultdict
from typing import Iterator, List, Tuple, Dict
from numpy import dtype

from pytorch_lightning import LightningDataModule
from torch import LongTensor, randperm, stack, unique, cat
from torch.utils.data import DataLoader, Sampler, SequentialSampler, Subset, random_split
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor, RandomAffine, ColorJitter)
from random import choice

from data.utils import get_sorted_dataset

import torch


class CifarBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool, num_tasks: int, tasks_idx: Dict) -> None:
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
        idx = [torch.tensor(idx, dtype=torch.long)[randperm(len(idx))].split(batch_size) for _, idx in tasks_idx.items()]
        idxlist = []
        for li in idx: idxlist.extend(li)
        idxs = torch.stack(idxlist, dim=0)
        self.idx = idxs[randperm(len(idxs))]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.idx

    def __len__(self) -> int:
        return len(self.idx)


class LightningCifarTasks(LightningDataModule):
    def __init__(self, batch_size, tasks:List[Tuple[int,int]], cifar=100,  data_dir=".data", num_class_per_task=10, n_classes=100, num_tasks=10, task_ids=None, color_jitter=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = Compose([RandomCrop(32, padding=4, padding_mode='reflect'),
                                  RandomHorizontalFlip(),
                                #   RandomAffine(degrees=30, scale=(.9, 1.1), shear=0),
                                  ToTensor(),
                                  Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.test_transform = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if color_jitter:
            self.transform = Compose([ColorJitter(hue=[0.2, 0.3]),
                            self.transform])
            self.test_transform = Compose([ColorJitter(hue=[0.2, 0.3]),
                            self.test_transform])
        self.num_class_per_task = num_class_per_task
        self.n_classes = n_classes
        self.dataset_gen = CIFAR100 if cifar == 100 else CIFAR10
        # self.classes = sum([list(classes) for classes in tasks])
        # if n_classes != num_tasks * num_class_per_task:
        #     raise ValueError()
        self.num_tasks = num_tasks
        if len(tasks) != num_tasks:
             raise ValueError()
        self.tasks = tasks
        self.task_ids = task_ids if task_ids is not None else list(range(len(tasks)))

    def prepare_data(self):
        self.dataset_gen(root=self.data_dir, train=True,
                 download=True, transform=ToTensor())
        self.dataset_gen(root=self.data_dir, train=False,
                 download=True, transform=ToTensor())

    def setup(self, stage):
        if stage == "fit" or stage is None:
            data_full = self.dataset_gen(root=self.data_dir, train=True,
                                 download=False, transform=self.transform)
            val_len = len(data_full)//10
            self.data_train, self.data_val = random_split(data_full, [len(data_full)-val_len, val_len])
            self.train_idx = self.get_tasks_idx(self.data_train)
            self.val_idx = self.get_tasks_idx(self.data_val)

        if stage == "test" or stage is None:
            self.data_test = self.dataset_gen(
                root=self.data_dir, train=False, download=False, transform=self.test_transform)
            self.test_idx = self.get_tasks_idx(self.data_test)
    
    def get_tasks_idx(self, dataset):
        classes_idx = defaultdict(list)
        for i, (x, y) in enumerate(dataset):
            classes_idx[y].append(i)

        max_len = max(len(classes_idx[k]) for k in classes_idx.keys())
        max_len = max_len - max_len%self.batch_size
        for k in classes_idx.keys():
            while(len(classes_idx[k]) < max_len):
                classes_idx[k].append(choice(classes_idx[k]))
        tasks_idx = defaultdict(list)
        for task in self.tasks:
            for cls in task:
                tasks_idx[task].extend(classes_idx[cls][:max_len].copy())
        return tasks_idx

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_sampler=self.batch_sampler(self.data_train, self.train_idx), collate_fn=self.get_collate_fn())

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_sampler=self.batch_sampler(self.data_val, self.val_idx), collate_fn=self.get_collate_fn())

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_sampler=self.batch_sampler(self.data_test, self.test_idx), collate_fn=self.get_collate_fn())

    def batch_sampler(self, dataset, tasks_idx):
        num_tasks = len(self.tasks)
        return CifarBatchSampler(SequentialSampler(dataset), batch_size=self.batch_size, drop_last=False, num_tasks=num_tasks, tasks_idx=tasks_idx)

    def get_collate_fn(self):

        def collate_fn(batch):
            classes = LongTensor([y for _, y in batch])
            unique_classes = sorted(unique(classes).tolist())
            for i, t in enumerate(self.tasks):
                if all(x in t for x in unique_classes):
                    task = torch.tensor([self.task_ids[i]], dtype=torch.long)
                    task_classes = t
            x = stack([x for x, _ in batch])
            y = LongTensor([task_classes.index(y) for _, y in batch])
            return x, y, classes, task
        return collate_fn
