import numpy as np

from nn.abstract import Optimizer, Model, Loss


class Momentum(Optimizer):

    def __init__(self, model: Model, alpha: float, beta: float):
        super().__init__(model, alpha)
        self.beta = beta

        # v is a dictionary of numpy arrays. Each
        # numpy array is the same shape as the weights of the corresponding layer
        self.v = dict()
        self.v_bias = dict()

    def step(self):
        for layer in self.model.layers:
            _id = layer.id()

            if not self.skip_weights_update(layer):
                if _id not in self.v:
                    self.v[_id] = np.zeros_like(layer.weights)
                self.v[layer.id()] = self.beta * self.v[layer.id()] - self.lr * layer.dl_w
                layer.weights += self.v[layer.id()]

            if not self.skip_bias_update(layer):
                if _id not in self.v_bias:
                    self.v_bias[_id] = np.zeros_like(layer.bias)
                self.v_bias[layer.id()] = self.beta * self.v_bias[layer.id()] - self.lr * layer.dl_b
                layer.bias += self.v_bias[layer.id()]




class Nesterov(Optimizer):

    def __init__(self, model: Model, loss: Loss, alpha: float, beta: float):
        """
        Nesterov Momentum optimizer.

        Args:
            model: Model, model to optimize
            loss: Loss, loss function
            alpha: float, learning rate
            beta: float, momentum coefficient
        """
        super().__init__(model, alpha)
        self.beta = beta
        self.loss = loss

        self.model_copy = None

        # v is a dictionary of numpy arrays. Each
        # numpy array is the same shape as the weights of the corresponding layer
        # self.v = [np.zeros_like(layer.weights) for layer in model.layers]
        self.v, self.v_bias = self.init()

    def init(self):
        v = dict()
        v_bias = dict()
        for i, layer in enumerate(self.model.layers):
            if layer.has_weights():
                v[i] = np.zeros_like(layer.weights)

            if layer.has_bias():
                v_bias[i] = np.zeros_like(layer.bias)

        return v, v_bias


    def look_ahead(self):
        """
        Compute the loss using the lookahead weights (i.e. W +
        and return the loss.
        """
        # Deep copy the current model to prevent modifying the original model
        self.model_copy = self.model.copy()
        # Get the input to the model
        x = self.model.get_model_input()
        # Get the target of input data
        loss_target = self.loss.target.copy()

        for i, layer in enumerate(self.model_copy.layers):
            if not self.skip_weights_update(layer):
                # for each layer in the copied model, update the weights by adding the lookahead values
                layer.weights += self.beta * self.v[i]

            if not self.skip_bias_update(layer):
                layer.bias += self.beta * self.v_bias[i]

            # run the forward pass
            x = layer.forward(x)

        # compute the loss using the new output
        self.loss(x, loss_target)
        return

    def step(self):
        # Run the forward pass of the model using the lookahead weights
        self.look_ahead()
        # Compute the gradient
        dl_y = self.loss.backward()
        # Back propagation
        self.model_copy.backward(dl_y)

        for i, layer in enumerate(self.model.layers):
            if not self.skip_weights_update(self.model_copy.layers[i]):
                self.v[i] = self.beta * self.v[i] - self.lr * self.model_copy.layers[i].dl_w
                layer.weights += self.v[i]

            if not self.skip_bias_update(self.model_copy.layers[i]):
                self.v_bias[i] = self.beta * self.v_bias[i] - self.lr * self.model_copy.layers[i].dl_b
                layer.bias += self.v_bias[i]
