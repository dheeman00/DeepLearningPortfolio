"""
Plan for Newton Optimizer:
1. Compute Gradient: Get the first derivative of the loss with respect to parameters.
2. Compute Hessian: Approximate the second derivatives by perturbing each parameter.
3. Step Method: Use the gradient and Hessian to update parameters with Newton's method.
4. Stability: Regularization and some type of invertability for more stability?.

"""
import numpy as np
from tqdm import trange

from nn.abstract import Optimizer, Model, Loss


class Newton(Optimizer):
    def __init__(self,
                 model: Model,
                 loss: Loss,
                 lr=0.01,
                 epsilon=1e-5,
                 regularization_strength=1e-4,
                 verbose=False):
        super().__init__(model, lr)
        self.loss = loss  # Loss function
        self.regularization_strength = regularization_strength  # Small num to stabilize Hessian inversion
        self.epsilon = epsilon  # Small num for finite difference

        self.shape_by_layer = self.model.shape()
        self.size_by_layer = self.model.size(reduction=False)
        self.hessian = None
        self.disable_progress_bar = not verbose

    def get_X(self):
        return self.model.get_model_input(copy=True)

    def get_y(self):
        return self.loss.get_target(copy=True)

    def retrieve_gradient(self, model: Model, flatten: bool = True):
        """
        Retrieve gradient from the input model

        Args:
            model: Model, model to retrieve gradient from.
            flatten: bool, whether to flatten the gradients

        Returns:
            gradients: list of numpy arrays, gradients for each layer
        """
        grads = [layer.dl_w.copy() for layer in model.layers]
        if flatten:
            return np.concatenate([grad.flatten() for grad in grads])
        return grads

    def compute_loss(self, model):
        """
        Compute the loss for a given model using the input data and target labels from the original model.

        Args:
            model: a copy of the original model with perturbed weights

        Returns:
            loss: float, loss value
        """
        X = self.get_X()
        y_true = self.get_y()
        y_pred = model(X)
        return self.loss(y_pred, y_true)

    def compute_gradient(self, model, flatten: bool = True):
        """
        For a given model, compute the gradient of the loss w.r.t. the parameters using
        the input data and target labels from the original model.

        Args:
            model: a copy of the original model with perturbed weights
            flatten: bool, whether to flatten the gradients

        Returns:
            gradients: list of numpy arrays, gradients for each layer
        """
        _ = self.compute_loss(model)
        loss_gradient = self.loss.backward()  # Get gradient of loss w.r.t. output
        model.backward(loss_gradient)  # Back propagate to get parameter gradients
        return self.retrieve_gradient(model=model, flatten=flatten)

    def locate_param(self, param_index):
        """
        Locate the parameter to perturb given an index

        Args:
            param_index: int, index of parameter
            layer_size: list, size of each layer
            layer_shape: list[tuple], shape of each layer

        Returns:
            layer: Layer, layer containing the parameter
            layer_index: int, index of the layer
            param_index: int, index of the parameter in the layer
        """
        layer_idx = 0
        while param_index >= self.size_by_layer[layer_idx]:
            param_index -= self.size_by_layer[layer_idx]
            layer_idx += 1

        # find row and column index
        layer_ncol = self.shape_by_layer[layer_idx][1]
        row_idx = param_index // layer_ncol
        col_idx = param_index % layer_ncol

        return layer_idx, row_idx, col_idx

    def compute_perturbed_gradient(self, model_copy, total_parameters):
        """
        Compute the perturbed gradient for each parameter in the model
        
        Args:
            model_copy: Model, model to compute the perturbed gradient
            total_parameters: int, total number of parameters in the model
            
        Returns:
            perturbed_gradients: numpy array, perturbed gradients for each parameter
        """
        perturbed_gradients = np.zeros((total_parameters, total_parameters))
        for i in trange(total_parameters, desc='Computing perturbed gradients', disable=self.disable_progress_bar):
            # Locate parameter i in model
            lid_i, rid_i, cid_i = self.locate_param(i)

            # store the original weight of i
            _weight_i = model_copy.layers[lid_i].weights[rid_i, cid_i]

            # Perturb parameter i
            model_copy.layers[lid_i].weights[rid_i, cid_i] += self.epsilon

            # compute the perturbed gradient wrt parameter i
            perturbed_gradients[i] = self.compute_gradient(model_copy, flatten=True)

            # restore the original weights i
            model_copy.layers[lid_i].weights[rid_i, cid_i] = _weight_i

        return perturbed_gradients

    def compute_hessian(self, initial_grad):
        initial_grad = initial_grad.reshape(-1, 1)  # Reshape to column vector
        model_copy = self.model.copy()
        total_parameters = sum(self.size_by_layer)  # Total params for Hessian size

        # pre-compute the perturbed gradient for each parameter
        perturbed_gradients = self.compute_perturbed_gradient(model_copy, total_parameters)

        # return the Hessian using finite-difference approximation
        # https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm
        return (perturbed_gradients - initial_grad + perturbed_gradients.T - initial_grad.T) / (2 * self.epsilon)

    def step(self):
        # Get the gradient
        grad_vector = self.retrieve_gradient(self.model)

        # Get the Hessian matrix
        self.hessian = self.compute_hessian(grad_vector)

        # Add regularization to Hessian for stability
        total_params = self.hessian.shape[0]
        self.hessian += self.regularization_strength * np.eye(total_params)

        # Invert the Hessian (I think there's a more robust way to do this)
        # Since we regularize we can use Cholesky decomposition if this doesn't work
        try:
            hessian_inv = np.linalg.inv(self.hessian)
        except np.linalg.LinAlgError:
            print("Hessian not invertible, skip this update ")
            return

        # Newton's method: delta = - H^{-1} * grad
        delta = - hessian_inv @ grad_vector

        # learning rate
        delta *= self.lr

        # Update the parameters
        # Flatten parameters
        params = np.concatenate([layer.weights.flatten() for layer in self.model.layers])

        # Update params again
        updated_params = params + delta

        # Reshape and assign updated parameters back to the model
        offset = 0
        for layer in self.model.layers:
            num_params = layer.weights.size
            layer.weights = updated_params[offset:offset + num_params].reshape(layer.weights.shape)
            offset += num_params
