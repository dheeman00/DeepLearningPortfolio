from nn.abstract import Model
from nn.abstract import Optimizer


class GD(Optimizer):

    def __init__(self, model: Model, lr: float = 0.001):
        super().__init__(model, lr)

    def step(self):
        for layer in self.model.layers:

            if not self.skip_weights_update(layer):
                layer.weights -= self.lr * layer.dl_w

            if not self.skip_bias_update(layer):
                layer.bias -= self.lr * layer.dl_b
