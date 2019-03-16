#!/usr/bin/python
# -*- encoding: utf8 -*-

import numpy as np


class Optimizer:
    def __init__(self):
        pass

    @staticmethod
    def gradient_descent(learning_rate, x, y, w, b):
        w = w + learning_rate * np.dot(y, x)
        b = b + learning_rate * y
        return w, b
