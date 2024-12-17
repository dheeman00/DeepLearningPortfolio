import numpy as np


def xavier_init(n_in, n_out, distribution='uniform'):
    """
    Xavier initialization for neural network weights.

    :param size: Shape of the weight matrix to initialize (e.g., (n_in, n_out)).
    :param n_in: Number of input units.
    :param n_out: Number of output units.
    :param distribution: 'uniform' or 'normal' for Xavier initialization.
    :return: Initialized weight matrix.
    """
    size = (n_in, n_out)

    if distribution == 'uniform':
        # Xavier uniform initialization
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(-limit, limit, size=size)
    elif distribution == 'normal':
        # Xavier normal initialization
        stddev = np.sqrt(2 / (n_in + n_out))
        return np.random.normal(0, stddev, size=size)
    else:
        raise ValueError("distribution must be 'uniform' or 'normal'")


def xavier_init_conv2d(fan_in, fan_out, shape, distribution='uniform'):
    """
    Xavier initialization for convolutional neural network weights.

    :param fan_in: Number of input units.
    :param fan_out: Number of output units.
    :param shape: Shape of the weight matrix to initialize (e.g., (out_channels, in_channels, kernel_size, kernel_size)).
    :param distribution: 'uniform' or 'normal' for Xavier initialization.
    :return: Initialized weight matrix.
    """
    if distribution == 'uniform':
        # Xavier uniform initialization
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)
    elif distribution == 'normal':
        # Xavier normal initialization
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=shape)
    else:
        raise ValueError("distribution must be 'uniform' or 'normal'")


# def pad2d_deprecated(x, witdh):
#     shape = x.shape
#     new_shape = (shape[0], shape[1], shape[2] + 2 * witdh, shape[3] + 2 * witdh)
#
#     new_x = np.zeros(new_shape)
#     new_x[..., witdh:-witdh, witdh:-witdh] = x
#     return new_x


def pad2d(x, width, pad_value=0):
    """
    Pad 2D input tensor. If the tensor is multi-dimensional, the padding will be applied to the last two dimensions.

    :param x: Input tensor of shape (N, C, H, W) or (N, H, W) or (H, W).
    :param width: Width of padding.
    :param pad_value: Padding value.
    :return: Padded tensor.
    """
    if width == 0:
        return x

    if len(x.shape) == 4:
        pad_width = ((0, 0), (0, 0), (width, width), (width, width))
    elif len(x.shape) == 3:
        pad_width = ((0, 0), (width, width), (width, width))
    elif len(x.shape) == 2:
        pad_width = ((width, width), (width, width))
    else:
        raise ValueError("Input tensor must be 2D, 3D or 4D.")

    return np.pad(x, pad_width, mode='constant', constant_values=pad_value)


def conv2d(x, kernel, stride=1, pad_width=0, pad_value=0, bias=None):
    """
    convolution for input array x and kernel.

    x is of shape (N, C, H_in, W_in), where
    - N is the number of samples,
    - C is the number of channels,
    - H_in is the height of the input,
    - W_in is the width of the input.

    kernel is of shape (F, C, H_k, W_k), where
    - F is the number of filters,
    - C is the number of channels that should match the number of channels in x,
    - H_k is the height of the kernel,
    - W_k is the width of the kernel.

    Args:
        x: Input array of shape (N, C, H_in, W_in).
        kernel: Kernel array of shape (F, C, H_k, W_k).
        stride: Stride of the convolution operation.
        pad_width: Width of the padding.
        pad_value: Value of the padding.
        bias: Bias array of shape (F,).

    Returns
        out: Output array of shape (N, F, H_out, W_out).

    """
    N, C, H_in, W_in = x.shape
    F, C, H_k, W_k = kernel.shape

    H_out = 1 + (H_in + 2 * pad_width - H_k) // stride
    W_out = 1 + (W_in + 2 * pad_width - W_k) // stride

    x_padded = pad2d(x, width=pad_width, pad_value=pad_value)

    out = np.zeros((N, F, H_out, W_out))

    # for n in range(N):
    #     for f in range(F):
    #         for i in range(0, H_out, stride):
    #             for j in range(0, W_out, stride):
    #                 window = x_padded[n, :, i:i + H_k, j:j + W_k]
    #                 out[n, f, i, j] = np.sum(window * kernel[f])
    #
    #         # add bias
    #         if bias is not None:
    #             out[n, f] += bias[f]

    # Optimized: skip the n loop
    for f in range(F):
        for i in range(0, H_out, stride):
            for j in range(0, W_out, stride):
                window = x_padded[:, :, i:i + H_k, j:j + W_k]
                out[:, f, i, j] = np.sum(window * kernel[f], axis=(1, 2, 3))

        # add bias
        if bias is not None:
            out[:, f] += bias[f]

    return out


# def rotate180_deprecated(kernel):
#     """
#     Rotate a 2D kernel by 180 degrees.
#
#     :param kernel: 2D kernel of shape (c_out, c_in, n, n).
#     :return: Rotated 2D kernel.
#     """
#     if len(kernel.shape) != 4:
#         raise ValueError("Input kernel must be 4D.")
#     return np.rot90(kernel, k=2, axes=(2, 3))


def rotate180(kernel):
    return kernel[..., ::-1, ::-1]
