from torch.nn import Module
from math import sqrt

class Selector(Module):
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