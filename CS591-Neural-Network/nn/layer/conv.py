import numpy as np

from nn.abstract import Activation
from nn.abstract import Layer
from nn.utilities import xavier_init_conv2d, conv2d, rotate180, pad2d


class Conv2d(Layer):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 activation: Activation,
                 stride: int = 1,
                 pad_width: int = 0
                 ):
        """
        Conv2d layer constructor.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the kernel.
            activation: Activation function.
            stride: Stride of the convolution operation.
            pad_width: Width of the padding.
            pad_value: Value of the padding.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.stride = stride
        self.pad_width = pad_width
        self.weights = self.xavier_init()
        self.bias = np.zeros(out_channels)

        self.h_prev = None
        self.a = None
        self.h = None

        self.dl_h_prev = None
        self.dl_a = None
        self.dl_w = None
        self.dl_b = None

    @property
    def filter(self):
        return self.weights

    def xavier_init(self):
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        fan_out = self.out_channels * self.kernel_size * self.kernel_size
        return xavier_init_conv2d(fan_in, fan_out,
                                  shape=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
                                  distribution='uniform')

    def forward(self, x):
        self.h_prev = x
        self.a = conv2d(x, self.filter, self.stride, self.pad_width, pad_value=0, bias=self.bias)
        self.h = self.activation(self.a)
        return self.h

    def backward(self, dl_y):
        """
        Backward pass for the Conv2d layer.
        Args:
            dl_y: shape (N, out_channels, H_out, W_out), gradient of the loss w.r.t the output of the layer
        Returns:

        """
        self.dl_a = dl_y * self.activation.backward(self.a)
        self.dl_w = self.compute_dl_f(self.dl_a)
        self.dl_b = self.compute_dl_b(self.dl_a)
        self.dl_h_prev = self.compute_dl_h_prev(self.dl_a)
        return self.dl_h_prev

    def compute_dl_f(self, dl_y):
        dl_w = np.zeros_like(self.filter)
        h_out, w_out = dl_y.shape[-2], dl_y.shape[-1]

        # Naive algorithm
        # for n in range(dl_y.shape[0]):
        #     for c_out in range(self.out_channels):
        #         _kernel = dl_y[n, c_out]  # shape (H_out, W_out)
        #         for c_in in range(self.in_channels):
        #             for i in range(self.kernel_size):
        #                 for j in range(self.kernel_size):
        #                     window = self.h_prev[n, c_in, i:i + _kernel.shape[0], j:j + _kernel.shape[1]]
        #                     dl_w[c_out, c_in, i, j] += np.sum(window * _kernel)

        # Optimized algorithm: skip the n loop
        # for c_out in range(self.out_channels):
        #     _kernel = dl_y[:, c_out]  # shape (H_out, W_out)
        #     h_out, w_out = _kernel.shape[1], _kernel.shape[2]
        #     for c_in in range(self.in_channels):
        #         for i in range(self.kernel_size):
        #             for j in range(self.kernel_size):
        #                 window = self.h_prev[:, c_in, i:i + h_out, j:j + w_out]
        #                 dl_w[c_out, c_in, i, j] += np.sum(window * _kernel)

        # Optimized: skip the n loop and the in_channels loop
        for c_out in range(self.out_channels):
            _kernel = dl_y[:, c_out]  # shape (H_out, W_out)
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    window = self.h_prev[:, :, i:i + h_out, j:j + w_out]
                    _conv = window * _kernel[:, None]
                    s = np.sum(_conv, axis=(0, 2, 3))
                    dl_w[c_out, :, i, j] += s

        return dl_w

    def compute_dl_b(self, dl_a):
        return np.sum(dl_a, axis=(0, 2, 3))

    def compute_dl_h_prev(self, dl_a):
        f_rotate = rotate180(self.filter)
        dl_y_pad = pad2d(dl_a, self.kernel_size - 1)

        dl_h_prev = np.zeros(self.h_prev.shape)  # shape (N, in_channels, H_in, W_in)

        for c_in in range(self.in_channels):
            _f = f_rotate[:, c_in]
            for i in range(self.h_prev.shape[-2]):
                for j in range(self.h_prev.shape[-1]):
                    window = dl_y_pad[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                    dl_h_prev[:, c_in, i, j] = np.sum(window * _f, axis=(1, 2, 3))
        return dl_h_prev

    def output_size(self):
        pass
