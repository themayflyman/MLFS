#!/usr/bin/python
# -*- encoding: utf8 -*-

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from Optimizer import Optimizer
from Dataset import IrisDataset


IRIS_DATASET = IrisDataset()


class Perceptron:
    def __init__(self):
        self.w = np.ones(1, dtype=np.float32)
        self.b = 0
        self._learning_rate = 0.1
        self._max_iteration = 1000

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
    def sign(x, w, b):
        y = np.dot(x, w) + b
        return y

    @staticmethod
    def cost_function(T, w, b):
        cost = 0
        for (x, y) in T:
            cost -= y * (np.dot(x, w) + b)
        return cost

    def _functional_margin(self, x, y):
        return y * self.sign(x, self.w, self.b)

    def fit(self, x_train, y_train):
        self.w = np.ones(x_train.shape[1])
        all_correct = False
        iteration = 0
        while not all_correct and iteration < self.max_iteration:
            wrong_count = 0
            for (x, y) in zip(x_train, y_train):
                if self._functional_margin(x, y) <= 0:
                    self.w, self.b = Optimizer.gradient_descent(self.learning_rate,
                                                                x, y, self.w, self.b)
                    wrong_count += 1
            if wrong_count == 0:
                all_correct = True
            iteration += 1

    def predict(self, x):
        y = self.sign(x, self.w, self.b)
        if y > 0:
            return 1
        else:
            return -1

    def score(self):
        pass

    @classmethod
    def test(cls):
        df = pd.DataFrame(IRIS_DATASET.data, columns=IRIS_DATASET.feature_names)
        df['label'] = IRIS_DATASET.target
        df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
        df.label.value_counts()
        data = np.array(df.iloc[:100, [0, 1, -1]])
        x_train, y_train = data[:, :-1], data[:, -1]
        y_train = np.array([1 if i == 1 else -1 for i in y_train])

        # training
        perceptron = cls()
        perceptron.fit(x_train, y_train)
        print(perceptron.w, perceptron.b)

        # testing
        x_points = np.linspace(4, 7, 10)
        y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
        plt.plot(x_points, y_)

        plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
        plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
        plt.legend()
        plt.show()
        cls.fit()


if __name__ == "__main__":
    Perceptron.test()
