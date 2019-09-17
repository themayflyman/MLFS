#!/usr/bin/python
# -*- encoding: utf8 -*-

from abc import ABCMeta, abstractmethod
import math
import copy

import numpy as np

from .activation_funcs import *


class Layer(metaclass=ABCMeta):
    layer_input = None
    _input_shape = None
    _output_shape = None

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape):
        self._input_shape = shape

    @property
    def output_shape(self):
        return self._output_shape

    @output_shape.setter
    def output_shape(self, shape):
        self._output_shape = shape

    @abstractmethod
    def forward_propagate(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward_propagate(self, *args, **kwargs):
        pass


class Dense(Layer):

    def __init__(self, num_units, input_shape=None):
        self.num_units = num_units
        self.input_shape = input_shape
        self.output_shape = (self.num_units, )

        bound = math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-bound, bound,
                                   (self.input_shape[0], self.num_units))
        self.W_optimizer = None

        self.b = np.zeros((1, self.num_units))
        self.b_optimizer = None

    def set_optimizers(self, optimizer):
        self.W_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

    def forward_propagate(self, layer_input):
        self.layer_input = layer_input
        return layer_input.dot(self.W) + self.b

    def backward_propagate(self, error):
        grad_wrt_W = self.layer_input.T.dot(error)
        grad_wrt_b = error

        error = error.dot(self.W.T)

        self.W = self.W_optimizer.update(self.W, grad_wrt_W)
        self.b = self.b_optimizer.update(self.b, grad_wrt_b)

        return error


class Flatten(Layer):
    """ turns a multidimensional matrix into two-dimensional"""

    def __init__(self, input_shape=None):
        self.actual_shape = None
        self.input_shape = input_shape

    def forward_propagate(self, X, training=True):
        self.actual_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_propagate(self, error):
        return error.reshape(self.actual_shape)

    @Layer.input_shape.setter
    def input_shape(self, shape):
        self._output_shape = (np.prod(self.input_shape),)
        self._input_shape = shape


class Dropout(Layer):
    """ A layer than randomly sets a fraction p of the output units of the
    the previous layer to zero.

    """
    def __init__(self, level, seed=None):
        # assert 0 < level < 1
        if not 0 < level < 1:
            raise ValueError('Dropout level must be in interval [0, 1).')
        self.level = level
        self.seed = seed
        self.r = None

    def forward_propagate(self, X, training=True):
        retain_prob = 1. - self.level
        if training:
            self.r = np.random.binomial(n=1, p=retain_prob, size=X.shape)
            X *= self.r
            X /= retain_prob

        return X

    def backward_propagate(self, error):
        return error * self.r

    @Layer.input_shape.setter
    def input_shape(self, shape):
        self._input_shape = shape
        self._output_shape = shape


activation_funcs = {
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'ReLU': ReLU,
    'leakyReLU': LeakyReLU,
    'tanH': TanH,
}


class Activation(Layer):

    def __init__(self, activation_func_name):
        self.activation_func = activation_funcs[activation_func_name]()

    def forward_propagate(self, layer_input):
        self.layer_input = layer_input
        return self.activation_func(layer_input)

    def backward_propagate(self, error):
        return self.activation_func.gradient(self.layer_input) * error


class Conv1D(Layer):
    pass


class Conv2D(Layer):
    pass


class PoolingLayer(Layer):
    def __init__(self, pool_size, stride=1, padding=0):
        pass

    def forward_propagate(self, *args, **kwargs):
        pass

    def backward_propagate(self, *args, **kwargs):
        pass


class MaxPooling1D(PoolingLayer):
    pass


class MaxPooling2D(PoolingLayer):
    pass


class AveragePooling1D(PoolingLayer):
    pass


class AveragePooling2D(PoolingLayer):
    pass

