from copy import deepcopy

from nn.abstract import Module


class Layer(Module):

    def __init__(self):
        super().__init__()
        # gradient of the loss w.r.t. the weights
        # (in_dim + 1, out_dim)
        self.dl_w = None

        self.weights = None
        self.h_prev = None

        self.bias = None
        self.dl_b = None

    def id(self):
        return id(self)

    def gd(self, lr):
        raise NotImplementedError

    def copy(self):
        """
        Create a copy of this layer
        """
        return deepcopy(self)

    def has_weights(self):
        return self.weights is not None

    def has_bias(self):
        return self.bias is not None

    def has_weight_grad(self):
        return self.dl_w is not None

    def has_bias_grad(self):
        return self.dl_b is not None
