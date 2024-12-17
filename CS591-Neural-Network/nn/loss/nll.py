import numpy as np

from nn.abstract import Loss


class NLLLoss(Loss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.log_sm = None  # output of the log softmax function
        self.sm = None  # output of the softmax function

    def forward(self, logsoftmax, target):
        """
        Compute the negative log likelihood loss.

        Args:
            logsoftmax: ndarray of shape (N, C). The predicted values, where N is the number of samples and C is the number of classes.
               it should be the output of the log softmax function.
            target: ndarray of shape (N,C). The one hot encoded target values.

        returns:
            loss: float. The negative log likelihood loss.
        """
        self.log_sm = self.ensure_ndarray(logsoftmax)
        self.sm = np.exp(logsoftmax)
        self.target = self.ensure_ndarray(target)

        _target = np.argmax(target, axis=1)
        loss = -self.reduction(logsoftmax[np.arange(len(target)), _target])
        return loss

    def backward(self):
        """
        Compute the gradient of the loss with respect to the input y.

        NOTE: since in the assignment, we always use NLLLoss with LogSoftmax, the gradient of the loss
        w.r.t. the pre-activation value is `o_i - y_i`, where o_i is the output of the SOFTMAX (not LogSoftMax) function
        and y_i is the target value. Therefore, we return this gradient directly. We will manually return a one matrix in
        the backward method of the LogSoftmax class.

        Returns:
            dl_y: ndarray of shape (N, C). The gradient of the loss with respect to y.
        """
        return (self.sm - self.target) / len(self.target)
