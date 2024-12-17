import numpy as np

from nn.abstract import Layer


class Flatten(Layer):
    def __init__(self, start_dim=0, end_dim=-1):
        """
        Initializes the Flatten class.
        Flattens input,which is given by call forward(), by reshaping it into a one-dimensional tensor. 
        If start_dim or end_dim are passed, only dimensions starting 
        with start_dim and ending with end_dim are flattened. 
        The order of elements in input is unchanged.
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.dim = None

    def forward(self, x) -> any:
        """
        Flattens the input array from start_dim to end_dim.
        Paras: x (numpy.ndarray): The array to flatten.
        Returns: numpy.ndarray: The flattened array.
        """
        self.dim = x.shape
        s_dim = self.start_dim
        e_dim = self.end_dim

        if s_dim < 0:
            s_dim = len(self.dim) + s_dim
        if e_dim < 0:
            e_dim = len(self.dim) + e_dim

        if s_dim > e_dim:
            raise ValueError("start_dim should be less than end_dim")

        if s_dim < 0:
            raise ValueError("start_dim should be greater than 0")

        if e_dim >= len(self.dim):
            raise ValueError("end_dim should be less than the dimension of the input array.")

        flattened_dim = np.prod(self.dim[s_dim:e_dim + 1])
        flattened_shape = self.dim[:s_dim] + (flattened_dim,) + self.dim[e_dim + 1:]

        return x.reshape(flattened_shape)

    def backward(self, dl_y) -> any:
        """
        In the backward pass, 
        it will reshape the gradient to the correct dimension 
        o that it can be passed into the convolutional layers
        
        Paras: dl_y (numpy.ndarray): The gradient array to reshape.
        
        Returns: numpy.ndarray: The reshaped gradient.
        """

        return dl_y.reshape(self.dim)
