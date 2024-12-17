from nn.abstract import Dataset


class BreastCancer(Dataset):
    def __init__(self, train_ratio: float = 0.8):
        super().__init__(train_ratio=train_ratio)
        self.X = None
        self.y = None
        self.data = None

    def shape(self):
        return self.X.shape

    def get(self):
        from sklearn.datasets import load_breast_cancer

        if self.data is None:
            self.data = load_breast_cancer()
            self.feature_names = self.data.feature_names
            self.target_names = self.data.target_names
            self.X = self.data.data
            self.y = self.data.target

        return self.X, self.y


if __name__ == '__main__':
    ds = BreastCancer()
    ds_train_X, ds_train_y = ds.get_train()
    ds_test_X, ds_test_y = ds.get_test()
