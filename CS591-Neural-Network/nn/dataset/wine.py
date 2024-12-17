from nn.abstract import Dataset


class Wine(Dataset):
    def __init__(self, binary_class_to_keep: int = None, train_ratio: float = 0.8):
        """
        Download the Wine dataset from sklearn.
        Args:
            binary_class_to_keep: if not None, it indicates the class we want to keep in the dataset.
                Example: `binary_class_to_keep=0` means we keep the first class, all the other classes will be merged into one class.
            train_ratio: the ratio of the training set
        """
        super().__init__(train_ratio=train_ratio)
        self.binary_class_to_keep = binary_class_to_keep
        self.X = None
        self.y = None
        self.data = None

    def shape(self):
        return self.data.data.shape

    def convert_to_binary(self, y, target_names):
        if self.binary_class_to_keep is not None:
            y = (y == self.binary_class_to_keep).astype(int)
            target_names = ['others', str(target_names[self.binary_class_to_keep])]

        return y, target_names

    def get(self):
        from sklearn.datasets import load_wine
        if self.data is None:
            self.data = load_wine()
            self.feature_names = self.data.feature_names
            self.X = self.data.data
            y = self.data.target
            target_names = self.data.target_names
            self.y, self.target_names = self.convert_to_binary(y, target_names)

        return self.X, self.y


if __name__ == '__main__':
    # Example of using the Wine dataset without converting to binary
    wine_original = Wine()
    wine_train_X, wine_train_y = wine_original.get_train()
    print(wine_original.target_names)
