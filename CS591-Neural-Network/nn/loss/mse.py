from nn.abstract import Loss


class MSELoss(Loss):
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.y = None

    def forward(self, y, target):
        self.y = self.ensure_ndarray(y)
        self.target = self.ensure_ndarray(target)

        loss = (self.target - self.y) ** 2
        return self.reduction(loss)

    def backward(self):
        return (2 / self.y.shape[0]) * (self.y - self.target)
