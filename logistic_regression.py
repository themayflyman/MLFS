#!/usr/bin/python
# -*- encoding: utf8 -*-

from math import exp, log2

import numpy as np
from dataset import IrisDataset

IRIS_DATASET = IrisDataset()


class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, max_iteration=200):
        self._learning_rate = learning_rate
        self._max_iteration = max_iteration
        self.weights = None
        self.classes = None

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def max_iteration(self):
        return self._max_iteration

    @max_iteration.setter
    def max_iteration(self, max_iteration):
        self._max_iteration = max_iteration

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def loss(self, weights, x_train, y_train):
        return -sum(y * log2(self.sigmoid(np.dot(weights, x))) +
                    (1 - y) * log2(1 - self.sigmoid(np.dot(weights, x)))
                    for x, y in zip(x_train, y_train)) / len(x_train)

    def fit(self, x_train, y_train):
        self.classes = set(y_train)
        self.weights = np.zeros(x_train.shape[1]+1, dtype=np.float32)
        for _ in range(self.max_iteration):
            for _x, _y in zip(x_train, y_train):
                # add a vector for b
                _x = np.array([1, *_x])
                self.weights += \
                    self.learning_rate * \
                    (_y - self.sigmoid(np.dot(_x, self.weights))) * _x.T

    def predict(self, x):
        result = np.dot(x, self.weights)
        if result > 0.5:
            return 1
        else:
            return 0

    def score(self):
        pass

    @classmethod
    def test(cls):
        x_train = IRIS_DATASET.data[:100, [0, 1]]
        y_train = IRIS_DATASET.target[:100]
        logistic_regression_clf = cls()
        logistic_regression_clf.fit(x_train, y_train)


if __name__ == "__main__":
    LogisticRegressionClassifier.test()
