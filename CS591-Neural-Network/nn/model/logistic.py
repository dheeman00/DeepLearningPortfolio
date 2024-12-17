import numpy as np

from nn.abstract import Module, ErrorLogger


class LogisticRegressionClassifier(Module, ErrorLogger):

    def __init__(self, dim: int):
        """
        Initialize the Logistic Regression Classifier.

        Parameters:
        dim (int): The dimensionality of the input data.
        """
        super().__init__()
        self.dim = dim  # Number of input features
        self.weights = None  # Weights for the model
        self.bias = None  # Bias for the model
        self.errors = []  # Stores errors for each epoch

        self.initialize()

    def initialize(self) -> None:
        """
        Initialize weights with small random values and bias to zero.
        Note: we increment the dimension of weight matrix by 1 to absorb the bias. We will not use/update bias
              in the model.
        """
        self.weights = np.random.uniform(-1, 1, size=self.dim + 1)
        self.bias = 0
        return

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the linear combination of input and weights.

        Parameters:
        x (np.ndarray): The input data (features).

        Returns:
        np.ndarray: The result of the linear combination plus the bias.
        """
        return np.dot(x, self.weights) + self.bias

    def activation(self, x):
        """
        Apply the sigmoid activation function.

        Parameters:
        x: The linear combination of input and weights.

        Returns:
        float: The sigmoid of the input, a probability value.
        """
        x = np.clip(x, -500, 500)  # add clipping to avoid overflow in sigmoid function
        return 1 / (1 + np.exp(-x))

    def delta_W(self, X, W, y):
        """
        Calculate the gradient of the loss with respect to the weights. (Lecture 7 page 3)

        Parameters:
            X: ndarray (n, k)
            W: ndarray (k,)
            y: ndarray (n,)

        Returns:
            scalar
        """
        nominator = y * X.T

        pre_value = y * W.reshape(-1, 1) * X.T  # add clipping to avoid overflow in exp()
        pre_value = np.clip(pre_value, -500, 500)
        denominator = 1 + np.exp(pre_value)

        # denominator = 1 + np.exp(y * W.reshape(-1, 1) * X.T)
        dw = - nominator / denominator
        return np.sum(dw.T, axis=0)

    def delta_W_v2(self, X, y, proba):
        """
        Gradient wrt W using probability (Lecture 7 page 3)
        """
        return np.sum((- proba * (y * X.T)).T, axis=0)

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

    def transform(self, X):
        """
        Predict the class labels based on the current weights and bias.

        Parameters:
        X (np.ndarray): The input data (features) of shape (n_samples, n_features).

        Returns:
        np.ndarray: The predicted class labels (-1 or 1) for each sample in X.
        """
        X = self.ensure_bias(X)
        z = self.forward(X)  # Compute the linear combination of inputs and weights
        post_act = self.activation(z)  # This is not used anymore due to the change to labels.
        y_pred = np.where(post_act >= 0.5, 1, -1)
        return y_pred

    def fit(self, X, y, epochs=5, lr=0.1):
        """
        Train the model using gradient descent.

        Parameters:
        X (np.ndarray): The input data (features) of shape (n_samples, n_features).
        y (np.ndarray): The true labels (-1 or 1) of shape (n_samples,).
        epochs (int): Number of epochs to train the model.
        lr (float): The learning rate for gradient descent.

        Returns:
        None
        """
        # Let the weight matrix absorb the bias
        X = self.ensure_bias(X)
        for epoch in range(epochs):
            # Calculate gradients
            dw = self.delta_W(X=X, W=self.weights, y=y)

            # calculate gradients with v2
            # (it seems that v1 works better)
            # _y_pred = self.transform(X)
            # proba = np.sum(_y_pred == y) / len(y)
            # dw = self.delta_W_v2(X, y, proba)

            # Update weights and bias
            self.weights -= lr * dw

            # Use transform() to make predictions
            y_pred = self.transform(X)

            # Count misclassified items
            misclassified_items = np.sum(y_pred != y)

            # Log the error (misclassified items)
            self.log_error(misclassified_items)
        return self


if __name__ == "__main__":
    from nn.dataset import BinaryDatasetGenerator, Wine, BreastCancer, Digits
    from nn.model import LogisticRegressionClassifier
    import numpy as np
    from nn.dataset.binary import case1

    X, y = BinaryDatasetGenerator(condition=case1, low=-1., high=1.).generate(n_pos=100, n_neg=100)

    lr = LogisticRegressionClassifier(dim=2)
    lr.fit(X, y, lr=0.01, epochs=25)
    print(lr.get_errors())

    # Wine dataset
    wine = Wine(binary_class_to_keep=0)
    X_train, y_train = wine.get_train()

    # convert y_train to -1 and 1
    y_train = np.array([1 if i == 1 else -1 for i in y_train])

    # normalize X_train
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    lr = LogisticRegressionClassifier(dim=X_train.shape[1])
    lr.fit(X_train, y_train, lr=0.001, epochs=50)
    print(lr.get_errors())
    print(f"final accuracy: {1 - lr.get_errors()[-1] / len(y_train)}")

    # Breast Cancer dataset
    bc = BreastCancer()
    X_train, y_train = bc.get_train()

    # convert y_train to -1 and 1
    y_train = np.array([1 if i == 1 else -1 for i in y_train])

    # normalize X_train
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    lr = LogisticRegressionClassifier(dim=X_train.shape[1])
    lr.fit(X_train, y_train, lr=0.001, epochs=50)
    print(lr.get_errors())
    print(f"final accuracy: {1 - lr.get_errors()[-1] / len(y_train)}")

    # Digit dataset
    digit = Digits(binary_class_to_keep=0)
    X_train, y_train = digit.get_train()

    # convert y_train to -1 and 1
    y_train = np.array([1 if i == 1 else -1 for i in y_train])

    # normalize X_train
    X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)

    lr = LogisticRegressionClassifier(dim=X_train.shape[1])
    lr.fit(X_train, y_train, lr=0.001, epochs=50)
    print(lr.get_errors())
    print(f"final accuracy: {1 - lr.get_errors()[-1] / len(y_train)}")
