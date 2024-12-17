from copy import deepcopy

import numpy as np

from nn.abstract import Module


class Loss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.target = None

        if reduction == 'none':
            self.reduction = lambda x: x
        elif reduction == 'mean':
            self.reduction = np.mean
        elif reduction == 'sum':
            self.reduction = np.sum
        else:
            raise ValueError(f'Unknown reduction type: {reduction}')

    @staticmethod
    def ensure_ndarray(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    def get_target(self, copy=True):
        if copy:
            return self.target.copy()
        else:
            return self.target

    def copy(self):
        """
        Create a copy of this loss
        """
        return deepcopy(self)
