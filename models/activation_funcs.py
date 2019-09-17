#!/usr/bin/python
# -*- encoding: utf8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


class ActivationFunc(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def gradient(self, *args, **kwargs):
        pass


class Sigmoid(ActivationFunc):

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax(ActivationFunc):

    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        pass

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class ReLU(ActivationFunc):

    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class LeakyReLU(ActivationFunc):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha*x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)


class TanH(ActivationFunc):

    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)
