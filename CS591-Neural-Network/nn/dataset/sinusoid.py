from nn.abstract import Dataset
import numpy as np
# Use type hints to help with any typing issues.
from typing import Tuple

class Sinusoidal(Dataset):
    def __init__(self, n_samples: int = 1000, x_range: Tuple[float, float] = (-3, 3), train_ratio: float = 0.8):
        super().__init__(train_ratio)
        self.n_samples = n_samples
        self.x_range = x_range
        self.X = None
        self.Y = None

    def shape(self) -> Tuple[int, int]:
        if self.X is not None:
            return self.X.shape
        else:
            return (0, 0)

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X is None:
            self._generate_data()
        return self.X, self.y

    def _generate_data(self) -> None:
        """
        Generate sin data with random samples.
        """
        # Generate random samples of x in the given range
        X = np.random.uniform(self.x_range[0], self.x_range[1], self.n_samples).reshape(-1, 1)
        # Generate corresponding y values as sin(x)
        y = np.sin(X)

        # Set the generated data to the instance attributes
        self.X = X
        self.y = y.flatten()  # Flatten y to make the shape (n_samples,)

        # Set feature and target names
        self.feature_names = ['x']
        self.target_names = ['sin(x)']

# Example test
if __name__ == "__main__":
    sin_dataset = Sinusoidal(n_samples=500)
    sin_dataset.train_test_split()
    X_train, y_train = sin_dataset.get_train()
    X_test, y_test = sin_dataset.get_test()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
