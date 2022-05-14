from collections import defaultdict
from typing import Iterator, List, Tuple, Dict
from numpy import dtype

from pytorch_lightning import LightningDataModule
from torch import LongTensor, randperm, stack, unique, cat
from torch.utils.data import DataLoader, Sampler, SequentialSampler, Subset, random_split, Dataset, ConcatDataset
from torchvision.datasets import CIFAR100, CIFAR10, MNIST
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor, RandomAffine, ColorJitter)
from random import choice

from data.utils import get_sorted_dataset

import torch
from tqdm import tqdm


class MNISTBatchSampler(Sampler[List[int]]):
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


class PermutedMNIST(Dataset):
    def __init__(self, root, train, permutation, task_id, transform=None) -> None:
        super().__init__()
        self.permutation = permutation
        self.task_id = task_id
        self.data = MNIST(root=root, train=train, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x, y = self.data[index]
        x_perm = x.view(-1)[self.permutation].view(1, 28, 28)
        return x_perm, y, y, torch.tensor(self.task_id, dtype=torch.int)


class LightningPermutedMNIST(LightningDataModule):
    def __init__(self, batch_size, data_dir=".data", num_tasks=10, task_ids=None):
        super().__init__()
        self.data_dir = data_dir
        self.root = data_dir
        self.batch_size = batch_size
        self.transform = Compose([ToTensor(),
                                  Normalize((0.485,), (0.229,))])
        self.test_transform = Compose([
            ToTensor(),
            Normalize((0.485,), (0.229,))])
        self.num_tasks = num_tasks
        self.dataset_gen = MNIST

    def prepare_data(self):
        self.dataset_gen(root=self.data_dir, train=True,
                 download=True, transform=ToTensor())
        self.dataset_gen(root=self.data_dir, train=False,
                 download=True, transform=ToTensor())
        train_datasets = []
        test_datasets = []
        for i in tqdm(range(self.num_tasks)):
            train_dataset = PermutedMNIST(root=self.root, train=True, permutation=torch.randperm(28*28), task_id=i, transform=self.transform)
            test_dataset = PermutedMNIST(root=self.root, train=False, permutation=torch.randperm(28*28), task_id=i, transform=self.transform)
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
        
        self.train_dataset = ConcatDataset(train_datasets)
        self.test_dataset = ConcatDataset(test_datasets)

    def setup(self, stage):
        if stage == "fit" or stage is None:
            data_full = self.train_dataset
            val_len = len(data_full)//10
            self.data_train, self.data_val = random_split(data_full, [len(data_full)-val_len, val_len])
            self.train_idx = self.get_tasks_idx(self.data_train)
            self.val_idx = self.get_tasks_idx(self.data_val)

        if stage == "test" or stage is None:
            self.data_test = self.test_dataset
            self.test_idx = self.get_tasks_idx(self.data_test)
    
    def get_tasks_idx(self, dataset):
        tasks_idx = defaultdict(list)
        for i, (x, y, c, t) in enumerate(dataset):
            tasks_idx[t.item()].append(i)

        max_len = max(len(tasks_idx[k]) for k in tasks_idx.keys())
        max_len = max_len - max_len%self.batch_size
        for k in tasks_idx.keys():
            while(len(tasks_idx[k]) < max_len):
                tasks_idx[k].append(choice(tasks_idx[k]))
        for k in tasks_idx.keys():
            tasks_idx[k] = tasks_idx[k][:max_len]
        return tasks_idx

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_sampler=self.batch_sampler(self.data_train, self.train_idx))

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_sampler=self.batch_sampler(self.data_val, self.val_idx))

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_sampler=self.batch_sampler(self.data_test, self.test_idx))

    def batch_sampler(self, dataset, tasks_idx):
        return MNISTBatchSampler(SequentialSampler(dataset), batch_size=self.batch_size, drop_last=False, num_tasks=self.num_tasks, tasks_idx=tasks_idx)

    # def get_collate_fn(self):

    #     def collate_fn(batch):
    #         classes = LongTensor([y for _, y in batch])
    #         unique_classes = sorted(unique(classes).tolist())
    #         for i, t in enumerate(self.tasks):
    #             if all(x in t for x in unique_classes):
    #                 task = torch.tensor([self.task_ids[i]], dtype=torch.long)
    #                 task_classes = t
    #         x = stack([x for x, _ in batch])
    #         y = LongTensor([task_classes.index(y) for _, y in batch])
    #         return x, y, classes, task
    #     return collate_fn
