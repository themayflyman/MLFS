#!/usr/bin/python
# -*- encoding: utf8 -*-

import copy
import numpy as np
# from sympy import *
from abc import ABCMeta, abstractmethod
from losses import *


def gradient(func, params, dimension, **kwargs):
    delta = kwargs.get('delta', 1e-10)
    gfunc = kwargs.get('gfunc')

    if gfunc:
        return gfunc(params)
    else:
        return func(dict(params, dimension=params[dimension]+delta)) - func(params) / delta


def differentiate(func, params, **kwargs):
    delta = kwargs.get('delta', 1e-10)
    return func(params + delta) - func(params)


class Optimizer(metaclass=ABCMeta):
    def __init__(self, epochs=200):
        self.loss = None
        self._epochs = epochs
        # self._learning_rate = learning_rate

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, val):
        self._epochs = val

    @abstractmethod
    def minimize(self, loss, params, param_list):
        pass


class GradientDescentOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.param_and_grad = {}

    def compute_gradient(self, loss, params, param_list):
        for param_name in param_list:
            self.param_and_grad[param_name] = gradient(loss, params, param_name)

    @staticmethod
    def apply_gradient(params, param_and_grad):
        for param_name, grad in param_and_grad.items():
            params[param_name] -= grad

        return params

    # TODO: Use closure, whenever minimize() is called, it returns a
    #       minimization (an operation), all minimizations will be run only when
    #       run() is called
    def minimize(self, loss, params, param_list):
        # Define a minimization(a minimize operation?)
        # Use closure
        self.compute_gradient(loss, params, param_list)
        for _ in range(self.epochs):
            self.apply_gradient(params, self.param_and_grad)


class NewTonOptimizer(Optimizer):
    def minimize(self, loss, params, param_list):
        pass

    def run(self):
        pass


def newton_raphson_method(func, **kwargs):
    func_prime = kwargs.get("func_prime")
    initial_guess = kwargs.get("initial_guess", 0)
    solution = initial_guess

    # Would it be faster?
    # if func_prime:
    #     def step(x):
    #         return x - func(x) / func_prime(x)
    # else:
    #     def step(x):
    #         return x - func(x) / differentiate(func, x)

    while abs(func(solution)) > 1e-50:
        # step(solution)
        if func_prime:
            solution -= func(solution) / func_prime(solution)
        else:
            solution -= func(solution) / differentiate(func,solution)

    return solution


def sequential_minimal_optimize():
    pass

