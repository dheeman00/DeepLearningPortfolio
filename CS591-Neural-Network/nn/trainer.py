import time

import numpy as np

from nn.abstract import Model, Loss, Optimizer
from nn.optim import *


class SGD:
    def __init__(self,
                 model: Model,
                 loss: Loss,
                 X: np.ndarray,
                 y: np.ndarray,
                 batch_size: int = 0,
                 epochs: int = 100,
                 lr: float = 0.01,
                 snapshot_at: list[int] = None,
                 optimizer: Optimizer = None,
                 X_val: np.ndarray = None,
                 y_val: np.ndarray = None,
                 add_noise: bool = False,
                 verbose=False):
        """
        Initialize the SGD trainer

        NOTE: in our implementation, we randomly sample a batch of size `batch_size` from the training data,
        and consider it as an epoch. This is different from the traditional definition of an epoch where the entire
        training data is used once.

        Args:
            model (Model): The model to train
            loss (Loss): The loss function to use
            X (np.ndarray): The input data of shape (n_samples, n_features)
            y (np.ndarray): The target labels.
                            If binary classification, shape is (n_samples, 1), and the value should be -1 and 1.
                            If multiclass, shape is (n_samples, n_classes), where each row is a one-hot encoded vector.
            batch_size (int): The batch size for training, set to 0 to use the entire input
            epochs (int): The number of epochs to train the model
            lr (float): The learning rate, default is 0.01. Note that it will be omitted if you provide an optimizer.
                        It will only be used when no optimizer is provided (the default optimizer is `nn.optim.GD`).
            snapshot_at (list[int]): The epochs to take snapshots of the model
            optimizer (str): The optimizer to use. If not provided, we use Gradient Descent
            verbose (bool): Whether to print training information
        """

        self.model: Model = model
        self.loss: Loss = loss
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.X_val: np.ndarray = X_val
        self.y_val: np.ndarray = y_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.losses = np.zeros(epochs)
        self.val_losses = np.zeros(epochs)
        self.snapshot_at = list() if snapshot_at is None else snapshot_at
        self.__snapshots = list()
        self.add_noise = add_noise

        self.best_model = None
        self.best_val_loss = np.inf

        self.verify()
        self.optimizer: Optimizer = GD(self.model, self.lr) if optimizer is None else optimizer

    def get_model_snapshots(self):
        """
        Get the model snapshots at the specified epochs
        """
        return self.__snapshots

    def save_snapshot(self):
        """
        Save the model snapshot
        """
        self.__snapshots.append(self.model.copy())

    def verify(self):
        """
        Verify the input data and target labels
        """
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Number of samples in input data and target labels must be equal")

        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)
            print("Implicitly reshaping the target labels to (n_samples, 1)")

        if self.y.ndim > 2:
            raise ValueError("Target labels must be either binary or multiclass")

    def load_batch(self):
        """
        Load the data and labels for the current batch given the batch size
        """

        # random select indices
        if self.batch_size == 0:
            X_batch = self.X
            y_batch = self.y

        else:
            indices = np.random.choice(self.X.shape[0], self.batch_size)
            X_batch = self.X[indices]
            y_batch = self.y[indices]

        if self.add_noise:
            X_batch += np.random.normal(0, 0.1, X_batch.shape)

        return X_batch, y_batch

    def save_loss(self, epoch, loss):
        """
        Save the loss for the current epoch
        """
        self.losses[epoch] = loss
        val_loss = self.eval_model()
        self.save_val_loss(epoch, val_loss)

    def get_losses(self):
        """
        Get the losses for each epoch
        """
        return self.losses

    def eval_model(self):
        """
        Evaluate the model on the validation data
        """
        if self.X_val is None or self.y_val is None:
            return np.inf

        model = self.model.copy()
        loss = self.loss.copy()

        # sample a batch from the validation data
        indices = np.random.choice(self.X_val.shape[0], self.batch_size)
        X_batch = self.X_val[indices]
        y_batch = self.y_val[indices]

        y_pred = model(X_batch)
        l = loss(y_pred, y_batch)

        if l <= self.best_val_loss:
            self.best_val_loss = l
            self.best_model = model

        return l

    def save_val_loss(self, epoch, loss):
        """
        Save the validation loss for the current epoch
        """
        self.val_losses[epoch] = loss

    def get_val_losses(self):
        """
        Get the validation losses for each epoch
        """
        return self.val_losses

    def fit(self):
        """
        Train the model using SGD
        """
        elapsed = 0
        for epoch in range(self.epochs):
            tic = time.time()

            # foward pass
            X_batch, y_batch = self.load_batch()
            y_pred = self.model(X_batch)
            loss = self.loss(y_pred, y_batch)

            # backward pass
            dl_y = self.loss.backward()
            self.model.backward(dl_y)
            self.optimizer.step()

            toc = time.time()
            epoch_elapsed = toc - tic
            elapsed += epoch_elapsed

            self.save_loss(epoch, loss)

            if epoch in self.snapshot_at:
                self.save_snapshot()

            if self.verbose:
                msg = f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss}"

                if self.X_val is not None and self.y_val is not None:
                    msg += f", Validation Loss: {self.val_losses[epoch]}"

                msg += f", Time (exclude validation): {epoch_elapsed:.2f}s"

                print(msg)

        print(f"Training completed in {elapsed:.2f}s")

    def transform(self, X):
        """
        Transform the input data using the trained model

        Args:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The transformed data
        """
        model = self.best_model if self.best_model is not None else self.model
        return model(X)

    def get_best_model(self):
        """
        Get the best model
        """
        return self.best_model
