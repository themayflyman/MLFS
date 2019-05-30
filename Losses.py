#!/usr/bin/python
# -*- encoding: utf8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    pass


class Loss:
    def __init__(self):
        pass


class SigmoidCrossEntropy(Loss):
    pass


def sigmoid_cross_entropy(y, y_hat):
    return
