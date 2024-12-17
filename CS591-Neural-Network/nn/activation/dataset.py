import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, train_ratio: float = 0.8):
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.train_ratio = train_ratio
        self.feature_names = None
        self.target_names = None

    def shape(self):
        raise NotImplementedError('shape not implemented')

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the dataset and the labels
        Returns:
            tuple[np.ndarray, np.ndarray]: The dataset (n, k) and the labels (n,)
        """
        raise NotImplementedError('get not implemented')

    def train_test_split(self) -> None:
        """
        Split the dataset into training and testing sets.
        """
        X, y = self.get()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=self.train_ratio)
        return

    def get_train(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the training dataset and the labels
        Returns:
            tuple[np.ndarray, np.ndarray]: The training dataset (n, k) and the labels (n,)
        """
        if self.X_train is None or self.y_train is None:
            self.train_test_split()

        return self.X_train, self.y_train

    def get_test(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the testing dataset and the labels
        Returns:
            tuple[np.ndarray, np.ndarray]: The testing dataset (n, k) and the labels (n,)
        """
        if self.X_test is None or self.y_test is None:
            self.train_test_split()

        return self.X_test, self.y_test
