import numpy as np

from nn.abstract import *
from nn.dataset import *
from nn.dataset.binary import case1


class WidrowHoffClassifier(Module, ErrorLogger):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.weights = None
        self.errors = []

        self.initialize()

    def initialize(self) -> None:
        """
        initialize the weight and bias from [-1,+1]
        To incorporate the bias b, the input vector x can be augmented by
        adding a constant 1, so that the bias is included in the weight vector.
        The input vector x becoms [x1,x2,...,xn,1], and the weight vector becomes
        [w1,w2,wn...,b]
        """
        self.weights = np.random.uniform(low=-1., high=1., size=self.dim + 1)

    def activation(self, x):
        """ Linear activation function
        Args:
            x (np.ndarray): The input to the activation function
        Returns:
            np.ndarray: The output of the activation function
        """
        return x

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the dot product of the input and the weights, add the bias, and return the result
        """
        X = np.dot(X, self.weights)
        return self.activation(X)

    def delta_W(self, X, y_true, y_pred, lr):
        # encode the gradient of the loss function wrt to W
        return lr * X.T.dot(y_true * (1 - y_true * y_pred))

    def transform(self, X):
        """transform X to y based on current weight matrix,
           then apply a threshold for binary classification (1 or -1)
        """
        # Add a bias term (1) to the input data X for final prediction
        z = np.ones((X.shape[0], 1))
        X_bias = np.append(X, z, axis=1)

        # Get the raw predictions
        raw_predictions = self.forward(X_bias)

        # Apply a threshold (e.g., 0)
        return np.where(raw_predictions >= 0, 1, -1)  # Binary classification (1 or -1)

    def fit(self, X, y, epochs=50, lr=0.01, verbose=False):
        """
        Fit training data using Widrow-Hoff learning algorithm
            X (np.ndarray): {array-like}, shape = [n_samples, n_features]
            where n_samples is # of samples and
            n_features is # of features

            y (np.ndarray): Target values, [n_samples]
            epochs (int): The number of epochs to train the perceptron
            lr (float): The learning rate
            verbose (bool): If True, print the weights and bias after each update
        Returns
        -------
        self : object
        """
        # Ensure X is 2D, otherwise convert it to 2d as column vectors
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Add a bias term (1) to the input data X for final prediction
        # The input vector x becoms [x1,x2,...,xn,1],
        z = np.ones((X.shape[0], 1))
        X_bias = np.append(X, z, axis=1)

        for i in range(epochs):
            y_pred = self.forward(X_bias)

            # get the error and store it
            # error is the # of misclassifications

            # y_pred_error stores the 1d prediction results after
            # transform X into WidrowHoffClassifier without updating
            # weight
            y_pred_error = self.transform(X)

            # becasue this is a binary classification(-1,1)
            # The element-wise addition for misclassfication will be 0
            # thus, error is the # of 0.
            error = np.count_nonzero(np.add(y_pred_error, y) == 0)
            self.errors.append(error)

            # update weights based on GD
            self.weights = self.weights + self.delta_W(X_bias, y, y_pred, lr)

            if verbose:
                # print the weights and bias
                print(f"Epoch: {i}:")
                print(f"error: {error}, weights: {self.weights[:, -1]}, bias: {self.weights[-1]}")
                print("=====================================")
        return self


def evaluation(y, y_pred):
    # return the # of correct classification
    return np.count_nonzero(np.add(y_pred, y) != 0)


def normalize_2d_columnwise(matrix1, matrix2):
    # normalization in column wise
    # store the dimension of two matrice
    num_row1 = matrix1.shape[0]
    num_row2 = matrix2.shape[0]
    # concatenate two matrix together
    matrix = np.concatenate((matrix1, matrix2), axis=0)

    normalized_matrix = matrix.copy()

    for i in range(matrix.shape[1]):
        col = matrix[:, i]
        if np.all(col == 0):
            # Leave the zero column unchanged
            continue
        else:
            # Normalize the column (example: min-max normalization to range [0, 1])
            col_min = np.min(col)
            col_max = np.max(col)
            if col_max != col_min:  # Avoid division by zero
                normalized_matrix[:, i] = (col - col_min) / (col_max - col_min)
    return normalized_matrix[:num_row1, :], normalized_matrix[:num_row2, :]


def main():
    print("\ntest on case1 from assignment 1, you can changed it manually to test on other 2 cases ")
    train_X, train_y = BinaryDatasetGenerator(case1, low=-10, high=10).generate(100, 100)
    test_X, test_y = BinaryDatasetGenerator(case1, low=-10, high=10).generate(100, 100)
    # train model
    lrs = [0.0001, 0.001, 0.01, 0.05, 0.1]
    epoches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 90]

    dim = train_X.shape[1]
    for lr in lrs:
        for epoch in epoches:
            wh = WidrowHoffClassifier(dim).fit(train_X, train_y, epochs=epoch, lr=lr)
            test_pred_y = wh.transform(test_X)
            accuracy = evaluation(test_y, test_pred_y)
            err = (accuracy / len(test_pred_y)) * 100
            print(
                f"Epoch: {epoch}\tlr: {lr:<10}Correct prediction: {accuracy}({len(test_pred_y)})\tAccuracy: {err:.2f}%")

    print("\ntest on BREASTCANCER ")
    # test on BreastCancer dataset
    ds = BreastCancer()
    ds_train_X, ds_train_y = ds.get_train()
    ds_test_X, ds_test_y = ds.get_test()
    # do normalization
    ds_train_X, ds_test_X = normalize_2d_columnwise(ds_train_X, ds_test_X)
    # any 0 in y will be converted as -1
    ds_train_y = np.where(ds_train_y == 1, 1, -1)
    ds_test_y = np.where(ds_test_y == 1, 1, -1)

    dim = ds_train_X.shape[1]
    for lr in lrs:
        for epoch in epoches:
            wh = WidrowHoffClassifier(dim).fit(ds_train_X, ds_train_y, epochs=epoch, lr=lr)
            ds_test_pred_y = wh.transform(ds_test_X)
            accuracy = evaluation(ds_test_y, ds_test_pred_y)
            err = (accuracy / len(ds_test_pred_y)) * 100
            print(
                f"Epoch: {epoch}\tlr: {lr:<10}Correct prediction: {accuracy}({len(ds_test_pred_y)})\tAccuracy: {err:.2f}%")

    print("\ntest on WINE ")
    # test on Wine dataset
    wine_binary = Wine(binary_class_to_keep=2)
    wine_train_X, wine_train_y = wine_binary.get_train()
    wine_test_X, wine_test_y = wine_binary.get_test()
    wine_train_X, wine_test_X = normalize_2d_columnwise(wine_train_X, wine_test_X)

    # any 0 in y will be converted as -1
    wine_train_y = np.where(wine_train_y == 1, 1, -1)
    wine_test_y = np.where(wine_test_y == 1, 1, -1)

    dim = wine_train_X.shape[1]
    for lr in lrs:
        for epoch in epoches:
            wh = WidrowHoffClassifier(dim).fit(wine_train_X, wine_train_y, epochs=epoch, lr=lr)
            wine_test_pred_y = wh.transform(wine_test_X)
            accuracy = evaluation(wine_test_y, wine_test_pred_y)
            err = (accuracy / len(wine_test_pred_y)) * 100
            print(
                f"Epoch: {epoch}\tlr: {lr:<10}Correct prediction: {accuracy:<2}({len(wine_test_pred_y):<2})\tAccuracy: {err:.2f}%")

    print("\ntest on Digits ")
    # test on Digits dataset
    digits = Digits(train_ratio=0.8, binary_class_to_keep=1)
    digits_train_X, digits_train_y = digits.get_train()
    digits_test_X, digits_test_y = digits.get_test()
    digits_train_X, digits_test_X = normalize_2d_columnwise(digits_train_X, digits_test_X)
    # any 0 in y will be converted as -1
    digits_train_y = np.where(digits_train_y == 1, 1, -1)
    digits_test_y = np.where(digits_test_y == 1, 1, -1)

    dim = digits_train_X.shape[1]
    for lr in lrs:
        for epoch in epoches:
            wh = WidrowHoffClassifier(dim).fit(digits_train_X, digits_train_y, epochs=epoch, lr=lr)
            digits_test_pred_y = wh.transform(digits_test_X)
            accuracy = evaluation(digits_test_y, digits_test_pred_y)
            err = (accuracy / len(digits_test_pred_y)) * 100
            print(
                f"Epoch: {epoch}\tlr: {lr:<10}Correct prediction: {accuracy}({len(digits_test_pred_y)})\tAccuracy: {err:.2f}%")


if __name__ == "__main__":
    main()
