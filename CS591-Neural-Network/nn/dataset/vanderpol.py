from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import odeint

from nn.abstract import Dataset


class VanDerPol(Dataset):
    def __init__(self,
                 n_samples: int = 1000,
                 step_size: float = 0.5,
                 x_range: Tuple[float, float] = (-3, 3),
                 train_ratio: float = 0.8):
        super().__init__(train_ratio)
        self.n_samples = n_samples
        self.x_range = x_range
        self.step_size = step_size
        self.X = None
        self.Y = None

    def shape(self) -> Tuple[int, int]:
        if self.X is not None:
            return self.X.shape
        else:
            return 0, 0

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X is None:
            self._generate_data()
        return self.X, self.y

    def vanderpol_eq(self, x, t):
        """
        Defines the Van der Pol differential equation.
        x1_dot = x2
        x2_dot = -x1 + (1 - x1^2) * x2
        """
        x1, x2 = x
        return [x2, -x1 + (1 - x1 ** 2) * x2]

    def _phi(self, x):
        """
        Compute one-step reachability of the VanderPol system.
        Args:
            x (array-like): Initial state of the system [x1, x2].
        Returns:
            array: Returns result after time step of 0.5 seconds.
        """
        t = np.linspace(0, self.step_size, 101)
        sol = odeint(self.vanderpol_eq, x, t)
        return sol[-1]

    def _generate_data(self) -> None:
        """
        Generate Van der Pol data which oscillates with random samples in range.
        """

        # Random sample generation
        x1_samples = np.linspace(self.x_range[0], self.x_range[1], self.n_samples)
        x2_samples = np.linspace(self.x_range[0], self.x_range[1], self.n_samples)
        x1_samples = np.repeat(x1_samples, self.n_samples).reshape(-1, 1)  # [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
        x2_samples = np.tile(x2_samples, self.n_samples).reshape(-1, 1)  # [1, 2, 3] -> [1, 2, 3, 1, 2, 3]

        # Combination of x1 and x2 as input pairs and result the y vals
        X = np.hstack((x1_samples, x2_samples))
        Y = np.array([self._phi(x) for x in X])

        self.X = X
        self.y = Y

        self.feature_names = ['x1', 'x2']
        self.target_names = ['y1', 'y2']

        # Debugging: Plot X (Input Data)
        # plt.figure(figsize=(8, 6))
        # plt.scatter(self.X[:, 0], self.X[:, 1], label="Input Grid (X1 vs X2)", alpha=0.7)
        # plt.xlabel('x1')
        # plt.ylabel('x2')
        # plt.title('Generated Input Data (X)')
        # plt.legend()
        # plt.show()

        # Debugging: Plot Y (Output Data)
        # plt.figure(figsize=(8, 6))
        # plt.scatter(self.y[:, 0], self.y[:, 1], label="Output (Y1 vs Y2)", alpha=0.7, color='orange')
        # plt.xlabel('y1')
        # plt.ylabel('y2')
        # plt.title('Generated Output Data (Y)')
        # plt.legend()
        # plt.show()

    # Maybe add a mini_batches method here

# Example
if __name__ == "__main__":
    from nn.dataset import VanDerPol

    vdp_dataset = VanDerPol(n_samples=101, step_size=10)
    vdp_dataset.train_test_split()
    X_train, y_train = vdp_dataset.get_train()
    X_test, y_test = vdp_dataset.get_test()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
