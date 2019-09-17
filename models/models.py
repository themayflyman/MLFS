#!/usr/bin/python
# -*- encoding: utf8 -*-

from copy import copy
import logging

import numpy as np

from .utils import shuffle_data, generate_batch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s")


class Model:

    def __init__(self, optimizer, loss_func, layers=None):
        self.optimizer = optimizer
        self.loss_func = loss_func()

        if layers:
            self.layers = layers
        else:
            self.layers = []

        self.logger = logging.getLogger(self.__name__)

    def add(self, layer):
        if not self.layers:
            layer.input_shape = self.layers[-1].output_shape

        if hasattr(layer, 'set_optimizers'):
            layer.set_optimizers(self.optimizer)

        self.layers.append(layer)

    def forward_propagate(self, layer_input, if_training=True):
        layer_output = layer_input
        for layer in self.layers:
            layer_output = layer.forward_propagate(layer_output, if_training)

        return layer_output

    def backward_propagate(self, error):
        for layer in reversed(self.layers):
            error = layer.backward_propagate(error)

    def fit_on_batch(self, X_batch, y_batch):
        y_batch_pred = self.forward_propagate(X_batch)

        error = self.loss_func.gradient(y_batch, y_batch_pred)
        self.backward_propagate(error)

    def evaluate_on_batch(self, X_batch, y_batch):
        y_batch_pred = self.predict_on_batch(X_batch)
        loss = np.mean(self.loss_func(y_batch, y_batch_pred))
        accuracy = 1 - np.count_nonzero(y_batch - y_batch_pred) / len(y_batch)
        # accuracy = np.sum(
        #     np.argmax(y_batch, axis=1) == np.argmax(y_batch_pred, axis=1),
        #     axis=0) / len(y_batch)

        return loss, accuracy

    def predict_on_batch(self, X_batch):
        return np.array([self.forward_propagate(x, if_training=False)
                         for x in X_batch])

    def fit(self, X, y, num_epochs, batch_size=32, shuffle=1, verbose=0):
        if verbose < 0:
            self.logger.setLevel(0)
        elif verbose == 0:
            self.logger.setLevel('INFO')
        else:
            self.logger.setLevel('DEBUG')
        for _ in range(num_epochs):
            logging.info("epoch{}".format(_))

            if shuffle:
                X, y = shuffle_data(X, y)

            for X_batch, y_batch in generate_batch(X, y, batch_size):
                self.fit_on_batch(X_batch, y_batch)
                loss, accuracy = self.evaluate_on_batch(X_batch, y_batch)

                self.logger.info(
                    "loss: {}, accuracy: {}".format(loss, accuracy))

    def evaluate(self, X, y, batch_size=32):
        pass

    def predict(self, X):
        return self.forward_propagate(X, if_training=False)


class Sequential(Model):
    def __init__(self, optimizer, loss):
        super().__init__(optimizer, loss)
