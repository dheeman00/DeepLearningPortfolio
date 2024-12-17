import numpy as np

from nn.abstract import Module


class Perceptron(Module):
    def __init__(self, input_size):
        super().__init__()
        self.weights = np.random.uniform(low=-1., high=1., size=input_size)
        self.bias = np.random.uniform(low=-1., high=1., size=1)

        # A list of epoch-wise errors. Each value represents the number of misclassified samples in each epoch.
        self.errors: list[float] = []

    def log_error(self, error):
        """
        Save the current error to track the performance of the perceptron
        """
        self.errors.append(error)

    def get_errors(self) -> list[float]:
        """
        Return the list of errors.

        Returns:
            list[float]: list of epoch-wise errors. Each value represents the number of misclassified samples in each epoch.
        """
        return self.errors

    def activation(self, x: np.ndarray) -> np.ndarray:
        """
        Activation function for the perceptron. For this implementation, we use the sign function.
        It returns 1 if x > 0, otherwise -1.

        Args:
            x (np.ndarray): The input to the activation function

        Returns:
            np.ndarray: The output of the activation function
        """
        return np.where(x > 0, 1, -1)

    def forward(self, x: np.ndarray):
        """
        Forward pass of the perceptron. 
        It computes the dot product of the input and the weights, adds the bias, and applies the activation function.
        """
        x = np.dot(x, self.weights)
        x += self.bias
        return self.activation(x)

    def fit(self, x, y, epochs=5, lr=0.1, verbose=False):
        """
        Fit the perceptron using the training data

        Args:
            x (np.ndarray): The training data
            y (np.ndarray): The training labels
            epochs (int): The number of epochs to train the perceptron
            lr (float): The learning rate
            verbose (bool): If True, print the weights and bias after each update
        """
        for e in range(epochs):
            error_count = 0

            for i in range(len(x)):
                # forward pass
                y_hat = self.forward(x[i])
                # compute the error
                error = y[i] - y_hat
                # update the weights and bias
                self.weights += lr * error * x[i]
                self.bias += lr * error

                # log the error count for analysis purpose
                _ec = int(np.sum(error != 0))
                error_count += _ec

            self.log_error(error_count)
            if verbose:
                # print the weights and bias
                print(f"Epoch: {e}:")
                print(f"error: {error_count}, weights: {self.weights}, bias: {self.bias}")
                print("=====================================")

    """
    Perception Model using Gradient Descent 
    """

    def fit_GD(self, x, y, epochs=200, lr=0.1, verbose=False):

        n_samples, n_features = x.shape
        # self.epochs = epochs
        # self.learning_rate = lr 
        # gradient descent
        for e in range(epochs):
            y_pred = self.forward(x)
            error = y_pred - y
            # calculate gradient for weight and bias
            dw = (2 / n_samples) * np.dot(x.T, error)
            db = (2 / n_samples) * np.sum(error)

            # update weights and bias
            self.weights -= lr * dw
            self.bias -= lr * db

            # log the error count for analysis purpose
            error_count = int(np.sum(error != 0))
            self.log_error(error_count)

            if verbose:
                # print the weights and bias
                print(f"Epoch: {e}:")
                print(f"error: {error_count}, weights: {self.weights}, bias: {self.bias}")
                print("=====================================")


if __name__ == '__main__':
    # example usage
    p = Perceptron(2)
    # print(perception.weights)    

    # test input
    x = np.array([
        [0.5, 0.5],
        [-0.1, 0.3],
        [0.8, -0.6],
        [-0.7, -0.8]])

    # test input label
    y = np.array([1, -1, 1, -1])

    # p.fit(x, y, epochs=10, lr=0.1, verbose=True)
    p.fit_GD(x, y, lr=0.1, epochs=20, verbose=True)
    p.get_errors()

    # predict the training data
    pred = p(x)
    print("=====================================")
    print(f'Predictions: {pred}')
    print(f'True labels: {y}')
