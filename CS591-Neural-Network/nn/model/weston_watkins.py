import numpy as np

from nn.abstract import Module, ErrorLogger


class WestonWatkinsSVM(Module, ErrorLogger):

    def __init__(self, n_feature: int, n_class: int):
        super().__init__()
        # init ErrorLogger
        ErrorLogger.__init__(self)

        self.n_feature = n_feature
        self.n_class = n_class
        self.weights = None

        self.initialize()

    def initialize(self) -> None:
        # Initialize the weights and bias
        self.weights = np.random.uniform(low=-1, high=1, size=(self.n_feature + 1, self.n_class))
        return

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Compute the dot product of the input and the weights, add the bias, and return the result
        return np.dot(x, self.weights)

    def activation(self, x):
        # Implement the activation function
        return x

    def delta_W(self, y_hat, y_true, x):
        """
        Compute the gradient of the loss function with respect to the weights

        Args:
            y_hat: ndarray of shape (n_samples, n_class), the predicted output of the model
            y_true: ndarray of shape (n_samples, n_class), the true labels (one-hot encoded)
            x: The input to the model

        Returns:
            gradient: ndarray of shape (n_features, n_classes), the gradient of the loss with respect to the weights
        """
        # compute the margin condition mask
        # for each prediction, get the true class prediction, ie. W_ci * X_i^T in slides
        true_class_y_hat = y_hat[y_true == 1].reshape(-1, 1)
        # compute margin for each prediction
        margin = np.maximum(0, y_hat - true_class_y_hat + 1)
        # set the margin to 0 for the true class
        margin[y_true == 1] = 0
        # compute the condition mask (ie. delta(j, Xi) in slides)
        cond_mask = (margin > 0).astype(int)
        # for the true class, sum the number of other classes that violate the margin
        cond_mask[y_true == 1] = np.sum(cond_mask, axis=1)

        # compute the class mask
        # for the true class, we want to increase the weight, thus +1,
        # for the other class, we want to decrease the weight, thus -1
        class_mask = np.where(y_true == 1, -1, 1)

        # combine the two masks
        mask = class_mask * cond_mask

        # calculate the gradient
        # x is of shape (n_samples, n_features), mask is of shape (n_samples, n_classes)
        # x.T @ mask is of shape (n_features, n_classes), which matches the shape of the weights
        gradient = x.T @ mask
        return gradient

    def ensure_shape(self, x):
        """ Ensure that the input is a 2D array """
        if x.ndim == 1:
            return x.reshape(1, -1)
        if x.ndim == 2:
            return x
        raise ValueError(f"Input shape {x.shape} is not supported")

    def ensure_bias(self, x):
        """ Ensure that the input has a bias term"""
        x = self.ensure_shape(x)
        if x.shape[1] == self.weights.shape[0]:
            return x
        if x.shape[1] == self.weights.shape[0] - 1:
            z = np.ones((x.shape[0], 1))
            x = np.append(x, z, axis=1)
            return x
        raise ValueError(f"Input shape {x.shape[1]} does not match the weight shape {self.weights.shape[0]}")

    def transform(self, x):
        # Add a bias term to the input if needed
        x = self.ensure_bias(x)
        x = self(x)
        x = self.activation(x)

        result = np.zeros_like(x)
        max_index = np.argmax(x, axis=1)
        result[np.arange(x.shape[0]), max_index] = 1
        return result

    def fit(self, x, y, epochs=5, lr=0.1, verbose=False):
        x = self.ensure_bias(x)

        for epoch in range(epochs):
            # Forward pass
            y_hat = self.forward(x)
            y_hat = self.activation(y_hat)
            # Compute the gradient
            gradient = self.delta_W(y_hat, y, x)
            # Update the weights
            self.weights -= lr * gradient

            # Compute the error
            y_pred = self.transform(x)
            errors = np.any(y_pred != y, axis=1)
            self.log_error(np.sum(errors))

        return


if __name__ == "__main__":
    from sklearn.preprocessing import OneHotEncoder
    from nn.dataset import Wine, Digits
    from nn.model import WestonWatkinsSVM
    import numpy as np

    # Load the Wine dataset
    wine = Wine()
    X, y = wine.get()

    # standardize the data
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    # One-hot encode the target labels
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

    # Create and train the model
    model = WestonWatkinsSVM(n_feature=X.shape[1], n_class=y.shape[1])
    model.fit(X, y, epochs=100, lr=0.01)
    print(model.get_errors())
    print(1 - model.get_errors()[-1] / len(y))

    # Load the Digits dataset
    digits = Digits()
    X, y = digits.get()

    # standardize the data
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    # One-hot encode the target labels
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

    # Create and train the model
    model = WestonWatkinsSVM(n_feature=X.shape[1], n_class=y.shape[1])
    model.fit(X, y, epochs=100, lr=0.01)
    print(model.get_errors())
    print(1 - model.get_errors()[-1] / len(y))
