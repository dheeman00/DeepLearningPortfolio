from nn.abstract import Activation
from nn.layer import Linear
from nn.model import Sequential


class FNN(Sequential):
    def __init__(self, dims: list[int], acts: list[Activation], initialization: str = 'xavier'):
        super().__init__()
        self.dims = dims
        self.acts = acts
        self.validate()

        for i in range(len(dims) - 1):
            layer = Linear(in_dim=dims[i], out_dim=dims[i + 1], act_func=acts[i], initialization=initialization)
            self.layers.append(layer)

    def validate(self):
        assert len(self.acts) == len(self.dims) - 1, 'Number of activations should be equal to number of dimensions - 1'

    def gd(self, lr):
        """
        NOTE: This method is deprecated. Use `nn.optim.GD` instead.
        Args:
            lr: learning rate

        Returns:

        """
        for layer in self.layers:
            layer.gd(lr)

    def size(self, reduction=True):
        """
        Return the number of parameters in the model

        Args:
            reduction: bool, whether to return the sum of all parameters or a list of sizes. Default is True
        Returns:
            int, number of parameters
        """
        sizes = [layer.weights.size for layer in self.layers]

        if reduction:
            return sum(sizes)
        return sizes

    def shape(self):
        """
        Return the shape of the model

        Returns:
            list of tuples, shape of each layer
        """
        return [layer.weights.shape for layer in self.layers]


if __name__ == "__main__":
    import numpy as np

    from nn.dataset import BreastCancer
    from nn.model import FNN
    from nn.activation import ReLU, Sigmoid
    from nn.loss import MSELoss
    from sklearn.preprocessing import StandardScaler

    data = BreastCancer()
    X_train, y_train = data.get_train()
    X_test, y_test = data.get_test()

    # standardize by row
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = np.where(y_train > 0, 1, -1).reshape(-1, 1)
    y_test = np.where(y_test > 0, 1, -1).reshape(-1, 1)

    fnn = FNN(dims=[X_train.shape[1], 10, 1], acts=[ReLU(), Sigmoid()])
    loss = MSELoss()

    # train
    for epoch in range(500):
        h = fnn.forward(X_train)
        y_pred = np.where(h > 0.5, 1, -1)
        l = loss(y_pred, y_train)
        accuracy = np.mean(y_pred == y_train)
        print(f'Epoch {epoch}, Loss {l}, Accuracy {accuracy}')

        dl_y = loss.backward()
        fnn.backward(dl_y)
        fnn.gd(lr=0.0001)

    # test
    h_test = fnn.forward(X_test)
    y_pred_test = np.where(h_test > 0.5, 1, -1)
    accuracy = np.mean(y_pred_test == y_test)
    print(f'Test accuracy {accuracy}')
