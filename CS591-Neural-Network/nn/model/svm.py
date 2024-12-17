import numpy as np

from nn.abstract import Module, ErrorLogger


class LinearSVM(Module, ErrorLogger):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.weights = None
        self.errors = []

        self.initialize()

    def initialize(self) -> None:
        """
        Initialize weights with small random values and bias to zero.
        Note: we increment the dimension of weight matrix by 1 to absorb the bias. We will not use/update bias
              in the model.
        """
        self.weights = np.random.uniform(-1, 1, size=self.dim + 1)
        return

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the linear combination of input and weights.

        Parameters:
        x (np.ndarray): The input data (features).

        Returns:
        np.ndarray: The result of the linear combination plus the bias.
        """
        return self.activation(np.dot(x, self.weights))

    def activation(self, x):
        """ Linear activation """
        return x

    def delta_W(self, y_hat, y_true, x, regularization: str = 'l2', lbd=0.):
        """
        Compute the gradient of the loss function with respect to the weights.
        """
        # mask the gradient if the prediction is correct
        # i.e. when y_hat * y_true >= 1, the mask will be 0, otherwise 1
        mask = (y_hat * y_true < 1).astype(int)

        if regularization == 'l2':
            regularization = lbd * self.weights
        elif regularization == 'l1':
            regularization = lbd * np.sign(self.weights)
        else:
            regularization = 0

        return np.dot(y_true * mask, x) + regularization

    def ensure_bias(self, x):
        """
        Ensure the input array has a bias term
        Args:
            x: 1d or 2d array
        Returns:
            x: 2d array with an additional bias term
        """
        # Ensure X is 2D, otherwise convert it to 2d as column vectors
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.ndim > 2:
            raise ValueError(f"Input shape {x.shape} is not supported. Only 1D and 2D arrays are supported.")

        # check the number of features in X,
        # if X is missing one feature, add a bias term
        if x.shape[1] != self.weights.shape[0]:
            if x.shape[1] == self.weights.shape[0] - 1:
                z = np.ones((x.shape[0], 1))
                x = np.append(x, z, axis=1)
            else:
                raise ValueError(f"Input shape {x.shape[1]} does not match the weight shape {self.weights.shape[0]}")
        return x

    def transform(self, x):
        """
        predict the class labels based on the current weights and bias.
        """
        x = self.ensure_bias(x)
        post_activ = self(x)
        y_pred = np.where(post_activ >= 0, 1, -1)
        return y_pred

    def fit(self, x, y, epochs=5, lr=0.1, verbose=False, regularization: str = 'l2', lbd=0.):
        """
        Fit the model to the data.

        Args:
            x: 1d or 2d array, input features
            y: 1d array, target labels
            epochs: int, number of iterations
            lr: float, learning rate
            verbose: bool
            regularization: str, regularization type, currently support L1 and L2
            lbd: float, regularization parameter
        """
        x = self.ensure_bias(x)
        for epoch in range(epochs):
            y_hat = self(x)
            dw = self.delta_W(y_hat=y_hat, y_true=y, x=x, regularization=regularization, lbd=lbd)
            # Update weights
            self.weights += lr * dw

            # log error
            pred = self.transform(x)
            error = np.sum(pred != y)
            self.log_error(error)


if __name__ == "__main__":
    from nn.dataset import BinaryDatasetGenerator, Wine, BreastCancer, Digits
    from nn.dataset.binary import case1
    from nn.model import LinearSVM
    import numpy as np

    X, y = BinaryDatasetGenerator(condition=case1, low=-1., high=1.).generate(n_pos=100, n_neg=100)

    model = LinearSVM(dim=2)
    model.fit(X, y, lr=0.01, epochs=25)
    print(model.get_errors())

    # Wine dataset
    wine = Wine(binary_class_to_keep=0)
    X_train, y_train = wine.get_train()

    # convert y_train to -1 and 1
    y_train = np.array([1 if i == 1 else -1 for i in y_train])

    # normalize X_train
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    model = LinearSVM(dim=X_train.shape[1])
    model.fit(X_train, y_train, lr=0.001, epochs=50, lbd=0.1)
    print(model.get_errors())
    print(f"final accuracy: {1 - model.get_errors()[-1] / len(y_train)}")

    # Breast Cancer dataset
    bc = BreastCancer()
    X_train, y_train = bc.get_train()

    # convert y_train to -1 and 1
    y_train = np.array([1 if i == 1 else -1 for i in y_train])

    # normalize X_train
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    model = LinearSVM(dim=X_train.shape[1])
    model.fit(X_train, y_train, lr=0.001, epochs=50, lbd=0.2)
    print(model.get_errors())
    print(f"final accuracy: {1 - model.get_errors()[-1] / len(y_train)}")

    # Digit dataset
    digit = Digits(binary_class_to_keep=0)
    X_train, y_train = digit.get_train()

    # convert y_train to -1 and 1
    y_train = np.array([1 if i == 1 else -1 for i in y_train])

    # normalize X_train
    X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)

    model = LinearSVM(dim=X_train.shape[1])
    model.fit(X_train, y_train, lr=0.001, epochs=50)
    print(model.get_errors())
    print(f"final accuracy: {1 - model.get_errors()[-1] / len(y_train)}")
