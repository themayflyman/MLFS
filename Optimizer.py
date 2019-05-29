#!/usr/bin/python
# -*- encoding: utf8 -*-

import copy
import numpy as np
from sympy import *
from abc import ABCMeta, abstractmethod
from Losses import *


class Optimizer(metaclass=ABCMeta):
    def __init__(self, epochs=200, learning_rate=0.01):
        self.loss = None
        self.epochs = epochs
        self.learning_rate = learning_rate

    @abstractmethod
    def minimize(self, loss, params, param_list):
        pass


class GradientDescentOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.param_and_grad = {}

    @staticmethod
    def gradient(func, params, dimension, **kwargs):
        delta = kwargs.get('delta', 1e-10)
        gfunc = kwargs.get('gfunc')

        if gfunc:
            return gfunc(params)
        else:
            return func(dict(params, dimension=params[dimension]+delta)) - func(params) / delta

    def compute_gradient(self, loss, params, param_list):
        for param_name in param_list:
            self.param_and_grad[param_name] = self.gradient(loss, params, param_name)

    @staticmethod
    def apply_gradient(params, param_and_grad):
        for param_name, grad in param_and_grad.items():
            params[param_name] -= grad

        return params

    def minimize(self, loss, params, param_list):
        self.compute_gradient(loss, params, param_list)
        for _ in range(self.epochs):
            self.apply_gradient(params, self.param_and_grad)


