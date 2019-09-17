#!/usr/bin/python
# -*- encoding: utf8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


class GradientDescentOptimizer(metaclass=ABCMeta):
    loss_func = None
    feed_dict = None
    history = None

    def feed(self, feed_dict):
        self.feed_dict = feed_dict

    @abstractmethod
    def update_w(self, w, grad_wrt_w):
        pass


class StochasticGradientDescentOptimizer(GradientDescentOptimizer):

    def __init__(self, learning_rate=0.1, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.history = {'w_updt': []}

    def update_w(self, w, grad_wrt_w):
        if not self.history['w_updt']:
            self.history['w_updt'].append(np.zeros(np.shape(w)))

        last_w_updt = self.history['w_updt'][-1]
        w_updt = self.momentum * last_w_updt + (1 - self.momentum) * grad_wrt_w
        self.history['w_updt'].append(w_updt)

        return w - self.learning_rate * w_updt


class AdagradOptimizer(GradientDescentOptimizer):

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.eps = 1e-8  # smoothing term
        self.sum_of_past_squared_grads = None

    def update_w(self, w, grad_wrt_w):
        if not self.G:
            self.sum_of_past_squared_grads = np.zeros(np.shape(w))

        self.sum_of_past_squared_grads += np.power(grad_wrt_w, 2)

        return w - self.learning_rate \
               / np.sqrt(self.sum_of_past_squared_grads + self.eps) * grad_wrt_w


class AdadeltaOptimizer(GradientDescentOptimizer):

    def __init__(self, rho=0.95, eps=1e-6):
        self.eps = eps
        self.rho = rho
        self.E_w_updt = None
        self.E_grad = None

    def update_w(self, w, grad_wrt_w):
        if not self.E_w_updt and not self.E_grad:
            self.E_w_updt = np.zeros(np.shape(w))
            self.E_grad = np.zeros(np.shape(grad_wrt_w))

        self.E_grad = self.rho * self.E_grad + (1 - self.rho) * np.power(grad_wrt_w, 2)

        rms_delta_w = np.sqrt(self.E_w_updt + self.eps)
        rms_grad = np.sqrt(self.E_grad + self.eps)

        w_updt = rms_delta_w / rms_grad * grad_wrt_w

        self.E_w_updt = self.rho * self.E_w_updt + (1 - self.rho) * np.power(w_updt, 2)

        return w - w_updt


class AdamOptimizer(GradientDescentOptimizer):

    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        # decay rates
        self.b1 = b1
        self.b2 = b2
        self.eps = 1e-8
        self.m = None
        self.v = None

    def update_w(self, w, grad_wrt_w):
        if not self.m:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        # decaying averages of past gradients
        self.m = self.b1 * self.m + (1 - self.b2) * grad_wrt_w
        # decaying averages of past squared gradients
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        bias_corrected_m = self.m / (1 - self.b1)
        bias_corrected_v = self.v / (1 - self.b2)

        w_updt = self.learning_rate * bias_corrected_m \
                 / (np.sqrt(bias_corrected_v) + self.eps)

        return w - w_updt


class AdaMaxOptimizer(GradientDescentOptimizer):

    def __init__(self, learning_rate, b):
        self.learning_rate = learning_rate
        self.b = b
        self.v = None

    def update_w(self, w, grad_wrt_w):
        if not self.v:
            self.v = np.zeros(np.shape(grad_wrt_w))

        u = np.max(self.b * self.v, np.abs(grad_wrt_w))

        self.v = self.b * self.v + (1 - self.b) *np.power(grad_wrt_w, 2)

        return w - u


