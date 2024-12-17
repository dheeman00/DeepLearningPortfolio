import numpy as np

from nn.abstract import Optimizer, Model


class Adam(Optimizer):

    def __init__(self,
                 model: Model,
                 lr: float,
                 rho_1: float,
                 rho_2: float,
                 bias_correction: bool = True,
                 epsilon: float = 1e-8):
        """
        Adam optimizer

        Args:
            model: model to optimize
            lr: learning rate
            rho_1: decay rate for the first moment estimate (F)
            rho_2: decay rate for the second moment estimate (A)
            epsilon: small value to avoid division by zero
        """
        super().__init__(model, lr)
        self.rho_2 = rho_2
        self.rho_1 = rho_1
        self.lr = lr
        self.bias_correction = bias_correction
        self.epsilon = epsilon
        self.a, self.a_bias = self.init_params()
        self.f, self.f_bias = self.init_params()
        self.t = 0

    def init_params(self):
        a = dict()
        a_bias = dict()
        for i, layer in enumerate(self.model.layers):
            if layer.has_weights():
                a[i] = np.zeros_like(layer.weights)

            if layer.has_bias():
                a_bias[i] = np.zeros_like(layer.bias)

        return a, a_bias

    def update_a(self, layer_idx: int):
        self.a[layer_idx] = self.rho_2 * self.a[layer_idx] + (1 - self.rho_2) * self.model.layers[layer_idx].dl_w ** 2

    def update_a_bias(self, layer_idx: int):
        self.a_bias[layer_idx] = (self.rho_2 * self.a_bias[layer_idx] +
                                  (1 - self.rho_2) * self.model.layers[layer_idx].dl_b ** 2)

    def update_f(self, layer_idx: int):
        self.f[layer_idx] = self.rho_1 * self.f[layer_idx] + (1 - self.rho_1) * self.model.layers[layer_idx].dl_w

    def update_f_bias(self, layer_idx: int):
        self.f_bias[layer_idx] = (self.rho_1 * self.f_bias[layer_idx] +
                                  (1 - self.rho_1) * self.model.layers[layer_idx].dl_b)

    def step(self):
        for i, layer in enumerate(self.model.layers):
            if not self.skip_weights_update(layer):
                self.update_a(i)
                self.update_f(i)
                if self.bias_correction:
                    self.t += 1

                    a_hat = self.a[i] / (1 - self.rho_2 ** self.t)
                    f_hat = self.f[i] / (1 - self.rho_1 ** self.t)
                else:
                    a_hat = self.a[i]
                    f_hat = self.f[i]

                layer.weights -= self.lr * f_hat / (np.sqrt(a_hat + self.epsilon))

            if not self.skip_bias_update(layer):
                self.update_a_bias(i)
                self.update_f_bias(i)
                if self.bias_correction:
                    a_bias_hat = self.a_bias[i] / (1 - self.rho_2 ** self.t)
                    f_bias_hat = self.f_bias[i] / (1 - self.rho_1 ** self.t)
                else:
                    a_bias_hat = self.a_bias[i]
                    f_bias_hat = self.f_bias[i]

                layer.bias -= self.lr * f_bias_hat / (np.sqrt(a_bias_hat + self.epsilon))
