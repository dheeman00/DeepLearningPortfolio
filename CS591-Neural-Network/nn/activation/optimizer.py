from nn.abstract import Model, Layer


class Optimizer:

    def __init__(self, model: Model, lr: float = 0.001):
        self.model = model
        self.lr = lr

    @staticmethod
    def verify(layer: Layer):
        if layer.dl_w is None:
            raise ValueError("Gradient of the loss w.r.t. the weights is not computed.")
        if layer.weights is None:
            raise ValueError("Weights are not initialized.")

    def step(self):
        raise NotImplementedError

    @staticmethod
    def skip_weights_update(layer: Layer):
        """
        Skip the weights update for this optimizer.
        """
        if layer.has_weights() and layer.has_weight_grad():
            return False

        return True

    @staticmethod
    def skip_bias_update(layer: Layer):
        """
        Skip the bias update for this optimizer.
        """
        if layer.has_bias() and layer.has_bias_grad():
            return False

        return True
