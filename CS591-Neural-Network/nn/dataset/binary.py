# -*- coding: utf-8 -*-
import inspect
from typing import Callable

import numpy as np


class BinaryDatasetGenerator:
    def __init__(self, condition: Callable[..., float], low, high, seed=42):
        """
        Binary dataset generator.

        Given a condition function that returns a float value, generate a binary dataset with positive and negative samples.

        The positive samples are the data points that satisfy the condition function (i.e., the output of the condition function is positive).
        The negative samples are the data points that do not satisfy the condition function (i.e., the output of the condition function is negative).

        Args:
            condition: a function that returns a float value
            low: lower bound of the random number
            high: upper bound of the random number
            seed: random seed
        """
        self.seed = seed
        np.random.seed(self.seed)

        self.low = low
        self.high = high
        self.condition = condition
        # get the number of arguments of the condition function
        self.n_args = len(inspect.signature(self.condition).parameters)

    def get_random_number(self):
        return np.random.uniform(self.low, self.high)

    def generate(self, n_pos: int = 100, n_neg: int = 100, shuffle: bool = True):
        """
        Generate a binary dataset with n_pos positive samples and n_neg negative samples.

        Args:
            n_pos: number of positive samples
            n_neg: number of negative samples
            shuffle: shuffle the dataset

        Returns:
            X: data
            y: labels
        """
        pos_samples = []
        neg_samples = []

        while len(pos_samples) < n_pos or len(neg_samples) < n_neg:
            # generate the arguments for the condition function
            args = [self.get_random_number() for _ in range(self.n_args)]

            # evaluate the condition function
            eval = self.condition(*args)

            # add the data to the positive sample if we have less than n_pos positive samples
            if eval > 0 and len(pos_samples) < n_pos:
                pos_samples.append(args + [1.0])

            # add the data to the negative sample if we have less than n_neg negative samples
            if eval < 0 and len(neg_samples) < n_neg:
                neg_samples.append(args + [-1.0])

        data = np.concatenate((pos_samples, neg_samples), axis=0)

        if shuffle:
            np.random.shuffle(data)

        x = data[:, :-1]
        # convert y to np.int
        y = data[:, -1].astype(int)

        return x, y


def case1(x1, x2):
    return -x1 + x2


def case2(x1, x2):
    return x1 - 2 * x2 + 5


def case3(x1, x2, x3, x4):
    return 0.5 * x1 - x2 - 10 * x3 + x4 + 50


if __name__ == "__main__":
    data_1, y_1 = BinaryDatasetGenerator(case1, low=-1, high=1).generate(100, 100)
    data_2, y_2 = BinaryDatasetGenerator(case2, low=-10, high=10).generate(100, 100)
    data_3, y_3 = BinaryDatasetGenerator(case3, low=-10, high=10).generate(100, 100)

    # general use case
    test_data, test_y = BinaryDatasetGenerator(lambda x1, x2: x1 - x2, low=-1, high=1).generate(10, 10)
