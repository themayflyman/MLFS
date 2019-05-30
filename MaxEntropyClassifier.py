#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
import itertools
import numpy as np
from Dataset import IrisDataset

IRIS_DATASET = IrisDataset()
EPSILON = 0.00001


def feature_func_example(feature, cls):
    def the_actual_feature_func(_feature, _cls):
        if feature == _feature and cls == _cls:
            return 1
        else:
            return 0
    return the_actual_feature_func


# TODO: detailed notes
class MaxEntropyClassifier:
    """introduction of this Max Entropy Classifier class*

    A conditional ME model, also known as a log linear model, has the following
    form:

                1        n
    P(y|x) = ------ exp(∑(w_i * f_i(x, y)))
              Z(x)      i=1

    where the functions f_i are the features of the model, the w_i are the
    parameters, or weights, and Z(x) is a normalisation constant. This form can
    be derived by choosing the model with maxinum entropy (i.e the most uniform
    model) from a set of models that satisfy a certain of constraints.

    >>> max_ent_clf = MaxEntropyClassifier(x_train, y_train)
    >>> max_ent_clf.train()
    >>>

    """
    def __init__(self, x_train, y_train, **kwargs):
        self.train_set = list(zip(x_train, y_train))
        self.train_set_num = sum(1 for _ in self.train_set)  # len(train_set)
        # functions that extracts feature
        self.features = set([_feature for _x in x_train for _feature in _x])
        self.feature_func = feature_func_example
        self.feature_funcs_example = dict((feature,
                                           dict((cls,
                                                 self.feature_func(feature,
                                                                   cls)
                                                 for cls in self.classes))
                                           for feature in self.features))
        self.feature_funcs = self.feature_funcs_example

        self.max_iteration = kwargs.get("max_iteration", 200)
        self.classes = kwargs.get("classes", set(y_train))
        self.Features = set(list(itertools.product(self.features,
                                                   self.classes)))
        self.weights = kwargs.get("weights",
                                  dict().fromkeys(self.Features, 0))

    @property
    def max_iteration(self):
        return self._max_iteration

    @max_iteration.setter
    def max_iteration(self, max_iteration):
        self._max_iteration = max_iteration

    @staticmethod
    def indicator_func(*args):
        for value_set in args:
            if value_set[0] != value_set[1]:
                return 0
        return 1

    def _compute_empirical_ffreq(self):
        empirical_ffreq = dict().fromkeys(self.features)
        for feature in self.features:
            empirical_ffreq[feature] = sum(
                [self.indicator_func((_feature, feature))
                 for _x, _y in self.train_set
                 for _feature in _x]
            )
        return empirical_ffreq

    def _compute_empirical_fcfreq(self):
        empirical_fcfreq = dict((feature, dict.fromkeys(self.classes)
                                 for feature in self.features))
        for feature in self.features:
            for cls in self.classes:
                empirical_fcfreq[feature][cls] = sum(
                    [self.indicator_func((_feature, feature), (_y, cls))
                     for _x, _y in self.train_set
                     for _feature in _x]
                )
        return empirical_fcfreq

    def _compute_empirical_expectation(self):
        return sum([self.empirical_fcproba[feature][cls] *
                    self.feature_funcs[feature][cls](feature, cls)
                    for feature in self.features for cls in self.classes])

    def _normalize(self, _x):
        return sum([
            math.exp(sum([self.weights[_feature] *
                          self.feature_funcs[_feature][cls](_feature, cls)
                          for _feature in _x]))
            for cls in self.classes])

    # TODO: modify the func name
    def _compute_Pyx(self, _x, _y):
        return math.exp(sum([
            self.weights[_feature] *
            self.feature_funcs[_feature][_y](_feature, _y)
            for _feature in _x])) / self._normalize(_x)

    def _compute_estimated_expectation(self):
        return sum([self.empirical_fproba[feature] *
                    self._compute_Pyx(feature, cls) *
                    self.feature_funcs[feature][cls](feature, cls)
                    for feature, cls in self.Features])

    def train(self, algorithm="IIS"):
        if algorithm == "IIS":
            self._train_max_ent_clf_with_iis()
        elif algorithm == "GIS":
            self._train_max_ent_clf_with_gis()
        elif algorithm == "BFGS":
            self._train_max_ent_clf_with_bfgs()

        # TODO: finish the return message of the max ent clf instance that is
        #       trained
        return "Max Entropy Classifier:\n" \
               "-max iteration: 200\n" \
               "-algorithm:{0}\n" \
               "" \
               "".format(algorithm)

    def _train_max_ent_clf_with_gis(self):
        """ Generalized Iterative Scaling

        """
        self.empirical_ffreq = self._compute_empirical_ffreq()
        self.empirical_fproba = dict(
            (feature, count / self.train_set_num
             for feature, count in self.empirical_ffreq.items()))
        self.empirical_fcfreq = self._compute_empirical_fcfreq()
        self.empirical_fcproba = dict(
            (feature, dict((cls, fccount / self.train_set_num
                            for cls, fccount in fccounts.items()))
             for feature, fccounts in self.empirical_fcfreq.items()
             ))

        def convergence(new_weights, old_weights):
            for new_weight, old_weight in zip(new_weights, old_weights):
                if abs(new_weight - old_weight) >= EPSILON:
                    return False
            return True

        # the empirical expected value of f_i
        empirical_expectation = self._compute_empirical_expectation()
        # the expected value of f_i according to model
        estimated_expectation = self._compute_estimated_expectation()

        # In theory C can be any constant greater than or equal to the figure in
        #
        #         C = MAX(SUM(f_i(x, y)))
        #             x,y
        #
        # In practice C is maximised over the (x,y) pairs in the training data,
        # since 1/C determines the rate of convergence of the algorithm it is
        # preferable to keep C as small as possible.
        c = max(set([len(_x) for _x, _y in self.train_set]))

        for _ in range(self.max_iteration):
            for feature in self.features:
                self.tmp_weights = self.weights
                self.weights[feature] += \
                    math.log(empirical_expectation / estimated_expectation) / c
                if convergence(self.weights, self.tmp_weights):
                    break

    # TODO: IIS Algorithm
    def _train_max_ent_clf_with_iis(self):
        """ Improved Iterative Scaling

        Steps:
            - Start with some (arbitrary) value for weight_feature
            - Repeat until convergence:
                          ∂B()
                -- Solve -------- = ∑ p(x, y)f_i(x, y) - ∑ p(x) ∑ p(y|x)f_i(x, y)
                         ∂(d_i)     x,y                  x

                -- Set w_i <-- w_i + d_i
        """
        pass

    # TODO: BFGS Algorithm
    def _train_max_ent_clf_with_bfgs(self):
        pass

    def classify(self, x):
        proba = {}
        for cls in self.classes:
            proba[cls] = self._compute_Pyx(x, cls)
        return max(proba, key=proba.get)

    @classmethod
    def test(cls):
        max_ent_clf = cls(IRIS_DATASET.data, IRIS_DATASET.target)
        max_ent_clf.train()


if __name__ == "__main__":
    MaxEntropyClassifier.test()
