from collections import defaultdict

from torch import LongTensor, stack, unique
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm


def get_sorted_dataset(dataset, batch_size, log=True, equalize=False):
    if log:
        print(f"Initial length: {len(dataset)}")
    indices = defaultdict(list)
    for idx, (x, y) in enumerate(tqdm(dataset)):
        indices[y].append(idx)
    for k in indices.keys():
        indices[k] = indices[k][:len(
            indices[k])-(len(indices[k]) % batch_size)]
    if equalize:
        min_nbr = min(len(indices[k]) for k in indices.keys())
        for k in indices.keys():
            indices[k] = indices[k][:min_nbr]
    datasets = [Subset(dataset, idx) for (cls, idx) in sorted(indices.items())]
    if log:
        print(f"{len(datasets)} classes")
    sorted_dataset = ConcatDataset(datasets)
    if log:
        print(f"Final length {len(sorted_dataset)}")

    return sorted_dataset


def get_collate_fn_task(num_class_per_task):
    def collate_fn(batch):
        x = stack([x for x, _ in batch])
        y = LongTensor([y % num_class_per_task for _, y in batch])
        classes = LongTensor([y for _, y in batch])
        task = LongTensor([y//num_class_per_task for _, y in batch])
        task = unique(task)
        if len(task) > 1:
            raise RuntimeError("There should be one task per batch")
        return x, y, classes, task
    return collate_fn
