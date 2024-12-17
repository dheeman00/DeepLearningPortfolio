from copy import deepcopy

from nn.abstract import Module, Layer


class Model(Module):
    def __init__(self):
        super().__init__()
        self.layers: list[Layer] = []

    def add_layer(self, layer: Layer):
        raise NotImplementedError

    def gd(self, lr):
        raise NotImplementedError

    def copy(self):
        return deepcopy(self)

    def get_model_input(self, copy=True):
        if copy:
            return self.layers[0].h_prev.copy()
        else:
            return self.layers[0].h_prev

    def size(self, reduction=True):
        """
        Return the number of parameters in the model

        Args:
            reduction: bool, whether to return the sum of all parameters or a list of sizes. Default is True
        Returns:
            int, number of parameters
        """
        raise NotImplementedError

    def shape(self):
        """
        Return the shape of the model

        Returns:
            list of tuples, shape of each layer
        """
        raise NotImplementedError
