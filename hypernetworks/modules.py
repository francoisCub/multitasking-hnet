from torch.nn import Module

class Selector(Module):
    def __init__(self, start, size, total_size=None):
        super().__init__()
        self.start = start
        self.end = start + size
        self.total_size = size

    def forward(self, x):
        return x[:, self.start:self.end]
    
    def extra_repr(self) -> str:
        return 'total_size={}'.format(
            self.total_size
        )