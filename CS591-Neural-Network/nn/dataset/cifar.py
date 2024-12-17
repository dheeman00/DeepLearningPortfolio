from nn.abstract import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from typing import Tuple
import matplotlib.pyplot as plt


# Uncomment the following if you have SSL verification error
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

class CIFAR10(Dataset):
    def __init__(self, train_ratio: float = -1.0):
        """Paras:train_ratio , if it is -1.0, which will keep 
        the original dataset split as 
        https://pytorch.org/vision/0.19/generated/torchvision.datasets.CIFAR10.html
        """
        super().__init__(train_ratio)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._generate_data()

    def _generate_data(self) -> None:
        """
        Load CIFAR10 data using torchvision, 
        normalize images to [-1, 1], 
        flatten them, 
        set up training and test data.
        """

        #get the training and testing dataset
        difar10_dataset_train = datasets.CIFAR10(root='./data',train=True,download=True)
        difar10_dataset_test = datasets.CIFAR10(root='./data',train=False,download=True)

        # Extract labelsX_test
        y_train = np.array(difar10_dataset_train.targets).astype(np.int64)
        y_test = np.array(difar10_dataset_test.targets).astype(np.int64)
 
        # Extract values, scale pixel values to [0, 1]
        X_train = difar10_dataset_train.data
        X_test = difar10_dataset_test.data

        #  (batch,H, W, C) transposes to (batch,C, H, W)
        X_train = np.transpose(X_train, (0, 3, 1, 2))
        X_test = np.transpose(X_test, (0, 3, 1, 2))

        # Scale the pixel values to [-1, 1]
        X_train = (X_train / 127.5) - 1
        X_test = (X_test / 127.5) - 1

        # Split into training and testing based on train_ratio
        # if train_ratio == -1.0, the split is based on the original dataset
        #otherwise, shuffle and split
        if self.train_ratio == -1.0:
            self.X_train, self.X_test, self.y_train, self.y_test = X_train,X_test,y_train,y_test
        else:
            X = np.vstack((X_test,X_train))
            y = np.concatenate((y_test,y_train))
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=self.train_ratio,shuffle=True)

    def get_train(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data has not been generated. Initialization problem")
        return self.X_train, self.y_train

    def get_test(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Testing data has not been generated. Initialization problem.")
        return self.X_test, self.y_test


    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X is None:
            raise ValueError("Dataset has not been generated. Problem with initialization")
        return self.X, self.y

    def shape(self, dataset_type: str = 'full') -> Tuple[int, int, int]:
        if dataset_type == 'train' and self.X_train is not None:
            return self.X_train.shape
        elif dataset_type == 'test' and self.X_test is not None:
            return self.X_test.shape
        elif dataset_type == 'full' and self.X is not None:
            return self.X.shape
        else:
            return (0, 0, 0)

    def get_class_name(self, class_id: int) -> str:
        return self.get_classes()[class_id]

    def get_classes(self):
        return ['plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



# Example usage
if __name__ == "__main__":
    dataset = CIFAR10()
    X_train, y_train = dataset.get_train()
    X_test, y_test = dataset.get_test()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Plot a few sample images to verify the dataset
    for i in range(3):
        # Transpose to (H, W, C) for visualization
        _img = (X_train[i] + 1) * 127.5
        image = np.transpose(_img.astype(int), (1, 2, 0))

        label = dataset.get_classes()[y_train[i]]
        plt.figure()
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.show()
