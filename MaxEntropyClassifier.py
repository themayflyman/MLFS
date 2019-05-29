#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
import numpy as np
from Dataset import IrisDataset

IRIS_DATASET = IrisDataset()
EPSILON = 0.00001


# TODO: detailed notes
class MaxEntropyClassifier:
    """
    """
    def __init__(self, x_train, y_train, **kwargs):
        self.feature_cls_pair = set([(_feature, _y)
                                     for _x, _y in zip(x_train, y_train)
                                     for _feature in _x])
        self.train_set = list(zip(x_train, y_train))
        self.train_set_num = sum(1 for _ in self.train_set)  # len(train_set)
        self.classes = set(y_train)
        # self.feature_labels = range(x_train.shape[1])
        self.features = set([_feature for _x in x_train for _feature in _x])
        # TODO: Figure out what feature functions really are
        self.feature_funcs = dict().fromkeys(self.features)

        # self.feature_dict = dict((feature_label,
        #                           dict().fromkeys(
        #                           set(x_train[:, feature_label])))
        #                          for feature_label in self.feature_labels)
        # self.empirical_fcount = dict().fromkeys(self.features)
        # self.empirical_fproba = dict(
        #     (feature, count / self.train_set_num
        #      for feature, count in self.empirical_fcount.items()))
        # self.empirical_fcproba = dict(
        #     (feature, dict((cls, fccount / self.train_set_num
        #                     for cls, fccount in fccounts.items()))
        #      for feature, fccounts in self.empirical_fccount.items()
        # ))
        # self.empirical_fccount = dict((feature, dict.fromkeys(self.classes)
        #                                for feature in self.features))
        # self.empirical_expectation = self._compute_empirical_expectation()

        self.max_iteration = kwargs.get("max_iteration", 200)
        self.weights = kwargs.get("weights",
                                  dict().fromkeys(self.features, 0))

    @property
    def max_iteration(self):
        return self._max_iteration

    @max_iteration.setter
    def max_iteration(self, max_iteration):
        self._max_iteration = max_iteration

    # @staticmethod
    # def feature_func(x, y, cond_func):
    #     return 1 if cond_func(x, y) else 0

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
        # return sum([self.empirical_fcprob[fc_pair] * self.feature_function(*fc_pair, cond_func=cond_func)
        #             for fc_pair in self.feature_cls_pair])
        return sum([self.empirical_fcproba[feature][cls] *
                    self.feature_funcs[feature](feature, cls)
                    for feature in self.features for cls in self.classes])

    def _normalize(self, _x):
        return sum([
            math.exp(sum([self.weights[_feature] *
                          self.feature_funcs[_feature](_feature, cls)
                          for _feature in _x]))
            for cls in self.classes])

    # TODO: modify the func name
    def _compute_Pyx(self, _x, _y):
        return math.exp(sum([
            self.weights[_feature] * self.feature_funcs[_feature](_feature, _y)
            for _feature in _x])) / self._normalize(_x)

    def _compute_estimated_expectation(self):
        return sum([self.empirical_fproba[feature] *
                    self._compute_Pyx(feature, cls) *
                    self.feature_funcs[feature](feature, cls)
                    for feature, cls in self.feature_cls_pair])

    def train(self, algorithm="IIS"):
        if algorithm == "IIS":
            self._train_max_ent_clf_with_iis()
        elif algorithm == "GIS":
            self._train_max_ent_clf_with_gis()
        elif algorithm == "BFGS":
            self._train_max_ent_clf_with_bfgs()

    def _train_max_ent_clf_with_gis(self):
        """ Generalized Iterative Scaling """
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

        empirical_expectation = self._compute_empirical_expectation()
        estimated_expectation = self._compute_estimated_expectation()

        # In theory C can be any constant greater than or equal to the figure in
        #
        #         C = MAX(SUM(f_i(x, y)))
        #             x,y
        #
        # In practice C is maximised over the (x,y) pairs in the training data,
        # since 1/C determines the rate of convergence of the algorithm it is
        # preferable to keep C as small as possible.
        c = max([sum([1 if feature in _x else 0 for _x, _y in self.train_set])
                 for feature in self.features])

        for _ in range(self.max_iteration):
            for feature in self.features:
                self.tmp_weights = self.weights
                self.weights[feature] += \
                    math.log(empirical_expectation / estimated_expectation) / c
                if convergence(self.weights, self.tmp_weights):
                    break

    # TODO: IIS Algorithm
    def _train_max_ent_clf_with_iis(self):
        """ Improved Iterative Scaling """
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



