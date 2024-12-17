from nn.abstract import Layer
import numpy as np


# The purpose of the pooling layer is to compress the input image.
# To do max pooling, each pooling window chooses the max val in its region.


class MaxPool2d(Layer):

    def __init__(self, kernel_size: int, stride: int = 0):
        super().__init__()
        # Size of the pooling window
        self.kernel_size = kernel_size
        # Default stride is kernel size if not provided
        self.stride = kernel_size if stride == 0 else stride
        # Binary mask for backward pass
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for max pooling.
        """
        if x.size == 0:
            raise ValueError("Input tensor is empty.")

        # Keep this, as useful for backward pass
        self.h_prev = x
        # Shape is usually:
        # Batch size: Amount of images at a time
        # Channels: 1 for greyscale, 3 for multicolor(higher the deeper)
        # Height: #of rows from input , Width: # of cols of inp
        batch_size, channels, height, width = x.shape
        kernel = self.kernel_size
        stride = self.stride

        # Get output dimensions.The equation is:
        # ((input_height/width - kernel_size)/stride)+1
        # Larger stride = smaller output dim
        output_height = (height - kernel) // stride + 1
        output_width = (width - kernel) // stride + 1

        # Init the output and mask
        pooled_output = np.zeros((batch_size, channels, output_height, output_width))
        self.mask = np.zeros_like(x)

        # Perform pooling over spatial dimensions
        for row in range(output_height):
            for column in range(output_width):
                row_start = row * stride
                row_end = row_start + kernel
                column_start = column * stride
                column_end = column_start + kernel

                # Get the pooling window
                pooling_window = x[:, :, row_start:row_end, column_start:column_end]

                # Vectorized max pooling over batch and channel
                pooled_output[:, :, row, column] = np.max(pooling_window, axis=(2, 3))

                # Create binary mask for max values
                mask = pooling_window == pooled_output[:, :, row, column][:, :, None, None]
                self.mask[:, :, row_start:row_end, column_start:column_end] = mask

        return pooled_output

    def backward(self, dl_y: np.ndarray) -> np.ndarray:
        # Basically the same thing as the forward although finding the max is diff and some
        # Other small things

        # Same as forward
        batch_size, channels, height, width = dl_y.shape
        kernel = self.kernel_size
        stride = self.stride

        # Make gradient for input contains 0's
        # This copies the shape of x
        dl_x = np.zeros_like(self.mask)

        # Backpropagate the gradient using the mask
        for row in range(height):
            for column in range(width):
                row_start = row * stride
                row_end = row_start + kernel
                column_start = column * stride
                column_end = column_start + kernel

                # Distribute the gradient to the corresponding input positions
                dl_x[:, :, row_start:row_end, column_start:column_end] += (
                        dl_y[:, :, row, column][:, :, None, None] *
                        self.mask[:, :, row_start:row_end, column_start:column_end]
                )
        return dl_x
