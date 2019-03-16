#!/usr/bin/python
# -*- encoding: utf8 -*-

from abc import ABCMeta, abstractmethod
from functools import reduce
import operator
import math
import numpy as np
import pandas as pd
from Dataset import WaterMelonDataset


WATERMELON_DATASET = WaterMelonDataset()


class BaseNaiveBayes(metaclass=ABCMeta):
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.classes = []
        self.feature_distributions = None
        self.prior_possibility = {}
        self.likelihood = {}

    @staticmethod
    def indicator_function(**kwargs):
        y = kwargs.get('y')
        cls = kwargs.get('cls')
        x = kwargs.get('x')
        feature = kwargs.get('feature')
        return 1 if y == cls and x == feature else 0

    def _compute_prior_possibility(self, y_train):
        for cls in self.classes:
            self.prior_possibility[cls] = \
                (sum(self.indicator_function(y=y, cls=cls)
                    for y in y_train) + self.alpha) / (len(y_train) + self.alpha * len(self.classes))

    @abstractmethod
    def _compute_likelihood(self, x_train, y_train):
        pass

    def fit(self, x_train, y_train, feature_distributions=None):
        self.feature_distributions = feature_distributions
        self.classes = set(y_train)
        self.prior_possibility.fromkeys(self.classes)
        self.likelihood = (dict((feature, dict.fromkeys(self.classes))
                                for feature in range(x_train.shape[1])))
        self._compute_prior_possibility(y_train)
        self._compute_likelihood(x_train, y_train)

    def predict(self, x):
        # For Python 3.7 and prior
        # an alternative for math.prod() in Python 3.8
        def prod(iterable):
            return reduce(operator.mul, iterable, 1)

        possibility = dict().fromkeys(self.classes)
        for cls in self.classes:
            # In Python 3.8, simply use use math.prod() instead
            possibility[cls] = self.prior_possibility[cls] * \
                               prod(self.likelihood[i][cls](x[i]) for i in range(len(x)))
        print(possibility)
        return max(possibility, key=possibility.get)


class GaussianNB(BaseNaiveBayes):
    def _compute_likelihood(self, x_train, y_train):

        def compute_likelihood_4_gaussian_feature(feature_train, cls):
            df = pd.DataFrame({'x': feature_train, 'y': y_train})
            avg = pd.to_numeric(df['x']).groupby(df['y']).mean().to_dict()[cls]
            std = pd.to_numeric(df['x']).groupby(df['y']).std().to_dict()[cls]
            print(avg, std)

            def gaussian_probability(feature_value):
                return (1 / (math.sqrt(2 * math.pi) * std)) * \
                    math.exp(-(math.pow(feature_value - avg, 2) /
                               (2 * math.pow(std, 2))))

            return gaussian_probability
        for cls in self.classes:
            for feature in range(x_train.shape[1]):
                self.likelihood[feature][cls] = \
                    compute_likelihood_4_gaussian_feature(x_train[:, feature],
                                                          cls)


class MultinomialNB(BaseNaiveBayes):
    def _compute_likelihood(self, x_train, y_train):
        def compute_likelihood_4_multinomial_feature(feature_train, y_train, cls):
            def multinomial_possibility(feature_value):
                return \
                    (sum(self.indicator_function(x=x, feature=feature_value,
                                                y=y, cls=cls) for x, y in zip(feature_train, y_train)) + self.alpha) / \
                    (sum(self.indicator_function(y=y, cls=cls) for y in y_train) + self.alpha * len(set(feature_train)))
                return t
            return multinomial_possibility
        for cls in self.classes:
            for feature in range(x_train.shape[1]):
                self.likelihood[feature][cls] = \
                    compute_likelihood_4_multinomial_feature(
                        x_train[:, feature], y_train, cls)


class BernoulliNB(BaseNaiveBayes):
    def _compute_likelihood(self, x_train, y_train):
        pass


class NaiveBayes(BaseNaiveBayes):
    def _compute_likelihood(self, x_train, y_train):
        for nb_model_cls, features in self.feature_distributions:
            nb_model = nb_model_cls(alpha=self.alpha)
            feature_train = x_train[:, features]
            nb_model.fit(feature_train, y_train)
            for feature, likelihood in zip(features, nb_model.likelihood.values()):
                self.likelihood[feature] = likelihood
            print(self.likelihood)


nb = NaiveBayes(alpha=0)
nb.fit(WATERMELON_DATASET.data, WATERMELON_DATASET.target, feature_distributions=[(GaussianNB, [6, 7]), (MultinomialNB, [0, 1, 2, 3, 4, 5])])
nb.predict(['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460])
