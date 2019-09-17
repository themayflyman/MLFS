#!/usr/bin/python
# -*- encoding: utf8 -*-

import numpy as np


def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


def generate_batch(X, y=None, batch_size=32):
    num_samples = X.shape[0]
    for i in np.arange(0, num_samples, batch_size):
        if y:
            yield X[i:i+batch_size], y[i:i+batch_size]
        else:
            yield X[i:i+batch_size]
