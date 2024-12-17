from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from nn.abstract import Dataset


class MNISTDataset(Dataset):
    def __init__(self, train_ratio: float = 0.8, size=(28, 28), flatten=True):
        super().__init__(train_ratio)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.size = size
        self.flatten = flatten
        self._generate_data()

    def shape(self, dataset_type: str = 'full') -> Tuple[int, int]:
        if dataset_type == 'train' and self.X_train is not None:
            return self.X_train.shape
        elif dataset_type == 'test' and self.X_test is not None:
            return self.X_test.shape
        elif dataset_type == 'full' and self.X is not None:
            return self.X.shape
        else:
            return (0, 0)

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X is None:
            raise ValueError("Dataset has not been generated. Problem with initialization")
        return self.X, self.y

    def _generate_data(self) -> None:
        """
        Load MNIST data using torchvision, normalize images to [-1, 1], flatten them, and set up training and test data.
        """
        mnist_data = datasets.MNIST(root="./data", train=True, download=True)

        transform = transforms.Compose([
            transforms.Resize(self.size),
        ])

        # Extract data and labels, flatten the images, and convert to numpy arrays
        X = transform(mnist_data.data).numpy().astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

        if self.flatten:
            X = X.reshape(-1, np.prod(self.size))
        else:
            # add channel dimension
            X = X[:, None, :, :]

        y = mnist_data.targets.numpy().astype(np.int64)

        # Split into training and testing based on train_ratio
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=self.train_ratio)

    def get_train(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data has not been generated. Initialization problem")
        return self.X_train, self.y_train

    def get_test(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Testing data has not been generated. Initialization problem.")
        return self.X_test, self.y_test

# Example usage
if __name__ == "__main__":
    mnist_dataset = MNISTDataset()
    X_train, y_train = mnist_dataset.get_train()
    X_test, y_test = mnist_dataset.get_test()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Plot a few sample images to verify the dataset
    for i in range(5):
        image = (X_train[i].reshape(28, 28) + 1) / 2  # Rescale to [0, 1] for visualization
        label = y_train[i]
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.show()

    # 3D data
    mnist_dataset = MNISTDataset(flatten=False, size=(32, 32))
    X_train, y_train = mnist_dataset.get_train()
    X_test, y_test = mnist_dataset.get_test()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Plot a few sample images to verify the dataset
    for i in range(5):
        image = (X_train[i].squeeze() + 1) / 2  # Rescale to [0, 1] for visualization
        label = y_train[i]
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.show()
