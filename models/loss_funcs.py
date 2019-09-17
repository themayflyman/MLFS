#!/usr/bin/python
# -*- encoding: utf8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


class LossFunc(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def gradient(self, *args, **kwargs):
        pass


class SquareLossFunc(LossFunc):

    def __call__(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)
