#!/usr/bin/python
# -*- encoding: utf8 -*-


import math
import numpy as np
from util import indicator_function


def adaptive_boosting(clf, x_train, y_train, m, **kwargs):
    n = x_train.shape[0]
    weights = kwargs.get("weights", np.ones(n) / n)
    learning_rate = kwargs.get("learning_rate", 0.5)
    f = lambda x: 0
    for _ in range(m):
        x_train = weights * x_train
        clf.train(x_train, y_train, sample_weight=weights)
        error = sum(w * indicator_function((clf.predict(x), y),
                                           cond=lambda a, b: 1 if a != b else 0)
                    for w, x, y in zip(weights, x_train, y_train))
        alpha = learning_rate * math.log(1 - error / error)
        regulator_factor = sum(w * math.exp(-alpha * y * clf.predict(x))
                               for w, x, y in zip(weights, x_train, y_train))
        for i, w, x, y in enumerate(zip(weights, x_train, y_train)):
            weights[i] = w * math.exp(-alpha * y * clf.predict(x)) / \
                         regulator_factor

        f = lambda x: f(x) + alpha * clf.decision_func(x)
    return lambda x: np.sign(f(x))


class BoostingTree:
    def __init__(self):
        pass
