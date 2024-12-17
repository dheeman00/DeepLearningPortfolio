from nn.abstract import Layer
from nn.abstract import Model


class Sequential(Model):

    def __init__(self):
        super().__init__()

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dl_y):
        for layer in reversed(self.layers):
            dl_y = layer.backward(dl_y)
