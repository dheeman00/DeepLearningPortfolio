from nn.model import Sequential
from nn.layer import Conv2d, MaxPool2d, Flatten, Linear
from nn.activation import ReLU, LogSoftmax


class CNN(Sequential):

    def __init__(self):
        super().__init__()


class LeNet5(CNN):

    def __init__(self, in_channel: int):
        super().__init__()
        self.add_layer(Conv2d(in_channel, 6, 5, activation=ReLU(), stride=1, pad_width=0))
        self.add_layer(MaxPool2d(2, 2))
        self.add_layer(Conv2d(6, 16, 5, activation=ReLU(), stride=1, pad_width=0))
        self.add_layer(MaxPool2d(2, 2))
        self.add_layer(Flatten(start_dim=1))
        self.add_layer(Linear(16 * 5 * 5, 120, act_func=ReLU()))
        self.add_layer(Linear(120, 84, act_func=ReLU()))
        self.add_layer(Linear(84, 10, act_func=LogSoftmax()))
