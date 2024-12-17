import numpy as np

from nn.abstract import Activation, Layer
from nn.utilities import xavier_init


class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int, act_func: Activation, initialization: str = 'xavier'):
        """
        Linear layer with activation function

        Args:
            in_dim: number of input features
            out_dim: number of output features
            act_func: activation function
            initialization: weight initialization method, one of ['xavier', 'random', 'zeros']
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = self.init_weights(initialization)
        self.act_func = act_func

        # partial derivative w.r.t the output (post-activation value) of the previous layer
        # (N, in_dim)
        self.dl_h_prev = None

        # partial derivative w.r.t the pre-activation value for this layer
        # (N, out_dim)
        self.dl_a = None

        # pre-activation value
        # (N, out_dim)
        self.a = None

        # post-activation value
        # (N, out_dim)
        self.h = None

        # input value
        # (N, in_dim + 1)
        self.h_prev = None

        # gradient of the loss w.r.t. the weights
        # (in_dim + 1, out_dim)
        self.dl_w = None

    def init_weights(self, init: str):

        if init == 'xavier':
            return xavier_init(self.in_dim + 1, self.out_dim, distribution='uniform')

        if init == 'random':
            return np.random.rand(self.in_dim + 1, self.out_dim)

        if init == 'zeros':
            return np.zeros((self.in_dim + 1, self.out_dim))

        if init == 'arange':
            return np.arange((self.in_dim + 1) * self.out_dim).reshape(self.in_dim + 1, self.out_dim)

        if init == 'ones':
            return np.ones((self.in_dim + 1, self.out_dim))

        raise ValueError(f"Initialization method {init} is not supported.")

    def ensure_bias(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure the input array has a bias term
        Args:
            x: 1d or 2d array
        Returns:
            x: 2d array with an additional bias term
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.ndim > 2:
            raise ValueError(f"Input shape {x.shape} is not supported. Only 1D and 2D arrays are supported.")

        # check the number of features in X,
        # if X is missing one feature, add a bias term
        if x.shape[1] != self.weights.shape[0]:
            if x.shape[1] == self.weights.shape[0] - 1:
                z = np.ones((x.shape[0], 1))
                x = np.append(x, z, axis=1)
            else:
                raise ValueError(f"Input shape {x.shape[1]} does not match the weight shape {self.weights.shape[0]}")
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Linear transformation followed by activation

        Args:
            x: input array (n_samples, in_dim)
        Returns:
            h: output after activation (n_samples, out_dim)
        """
        self.h_prev = self.ensure_bias(x)
        self.a = self.h_prev @ self.weights
        self.h = self.act_func(self.a)
        return self.h

    def backward(self, dl_h: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass.

        We use the partial derivative of loss w.r.t the post-activation value (dl_h)
        to compute the partial derivative of the loss w.r.t. the pre-activation value (dl_a) of this layer.
        Also, we compute the gradient of the loss w.r.t. the output of the previous layer (dl_h_prev)
        (i.e., input of this layer).

        Args:
            dl_h: gradient of the loss w.r.t. the output of this layer (n_samples, out_dim).
            It should be generated by the next layer.

        Returns:
            dl_h_prev: gradient of the loss w.r.t. the output of the previous layer (n_samples, in_dim)
        """

        self.dl_a = dl_h * self.act_func.backward(self.a)
        self.dl_h_prev = self.dl_a @ self.weights.T
        self.dl_w = self.h_prev.T @ self.dl_a
        return self.dl_h_prev[:, :-1]  # remove the bias term gradient

    def gd(self, lr: float):
        """
        NOTE: This method is deprecated. Use the step method in the optimizer instead.

        Perform gradient descent on the weights of this layer.

        Args:
            lr: learning rate

        Returns:
            dl_w: gradient of the loss w.r.t. the weights
        """
        self.weights -= lr * self.dl_w
        return self.dl_w
