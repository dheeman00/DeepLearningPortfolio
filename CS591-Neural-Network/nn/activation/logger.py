class ErrorLogger:
    def __init__(self):
        self.errors = []

    def log_error(self, error):
        """
        Save the current error to track the performance of the perceptron
        """
        self.errors.append(float(error))

    def get_errors(self) -> list[float]:
        """
        Return the list of errors.

        Returns:
            list[float]: list of epoch-wise errors. Each value represents the number of misclassified samples in each epoch.
        """
        return self.errors
