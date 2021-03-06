from math import sqrt

from torch import rand
from torch.nn import Module


class Selector(Module):
    """
        Selector module. Select a part of a target model parameters produced and rescaled them for proper initialization
    """
    def __init__(self, start, size, total_size=None, shape=None):
        super().__init__()
        self.start = start
        self.end = start + size
        self.total_size = size
        if len(shape) == 2:
            self.scaling_factor = sqrt(6 / (3*(shape[-1] + shape[-2])))
        elif len(shape) == 4:
            self.scaling_factor = sqrt(1 / (3*(shape[1]*shape[-1]*shape[-2])))
        else:
            self.scaling_factor = sqrt(1 / (3*shape[-1]))
        

    def forward(self, x):
        return x[:, self.start:self.end] * self.scaling_factor
    
    def extra_repr(self) -> str:
        return f'total_size={self.total_size}, scaling={self.scaling_factor:.4f}'


class Sparsify(Module):
    """
        Module that applies a binary mask on its input
    """
    def __init__(self, start, size, global_mask_vector, total_size=None, shape=None, mask_vector=None):
        super().__init__()
        self.size = size
        if mask_vector is None:
            mask_vector = global_mask_vector[start:start+size].float()
        else:
            mask_vector = mask_vector.float()
        self.register_buffer("mask_vector", mask_vector)
        self.actual_sparsity = self.mask_vector.mean().item()

    def forward(self, x):
        return x * self.mask_vector
    
    def extra_repr(self) -> str:
        return f'size={self.size}, actual_sparsity={self.actual_sparsity:.2f}'

def get_sparsify(total_size, sparsity):
    if sparsity > 1 or sparsity < 0:
        raise ValueError()
    global_mask_vector = rand(total_size) > sparsity
    def sparsify(start, size, total_size=None, shape=None):
        return Sparsify(start, size, global_mask_vector, total_size=total_size, shape=shape)
    return sparsify
