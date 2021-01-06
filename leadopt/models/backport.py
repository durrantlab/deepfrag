
from torch.nn import Module
from typing import Tuple, Union
from torch import Tensor

'''
On some older versions of Cuda, we may need to use an early version of PyTorch
that doesn't have the Flatten layer builtin.
'''

class Flatten(Module):
    __constants__ = ['start_dim', 'end_dim']
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return 'start_dim={}, end_dim={}'.format(
            self.start_dim, self.end_dim
        )
