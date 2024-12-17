class Module:
    def __init__(self):
        pass

    def forward(self, *args, **kwds) -> any:
        raise NotImplementedError('forward not implemented')

    def __call__(self, *args, **kwds) -> any:
        """
        This method allows you to call the object as a function.
        Example:
        >>> m = Module()
        >>> m(1)
        """
        return self.forward(*args, **kwds)

    def backward(self, *args, **kwds) -> any:
        raise NotImplementedError('backward not implemented')
