#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
import itertools
from Dataset import IrisDataset
from Optimizer import newton_raphson_method

IRIS_DATASET = IrisDataset()
EPSILON = 0.00001


def generate_feature_func(feature, cls):
    def feature_func(_feature, _cls):
        if feature == _feature and cls == _cls:
            return 1
        else:
            return 0
    return feature_func


# TODO: detailed notes
# TODO: Gaussian Prior Smoothing
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

    >>> max_ent_clf = MaxEntropyClassifier()
    >>> max_ent_clf.train(x_train=x_train, y_train=y_train)
    >>>
    >>>

    """
    def __init__(self, **kwargs):
        self.max_iteration = kwargs.get("max_iteration", 200)
        self.weights = kwargs.get("weights")
        self.classes = None
        self.train_set = None
        self.train_set_num = None
        self.features = None
        self.feature_funcs_example = None
        self.feature_funcs = self.feature_funcs_example
        self.fcs = None

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
        empirical_fcfreq = dict().fromkeys(self.fcs)

        for feature, cls in self.fcs:
            empirical_fcfreq[feature, cls] = sum(
                    [self.indicator_func((_feature, feature), (_y, cls))
                     for _x, _y in self.train_set
                     for _feature in _x]
                )

        return empirical_fcfreq

    def _compute_empirical_expectation(self):
        """To compute empirical expectation

        """
        return sum([self.empirical_fcprob[feature, cls] *
                    self.feature_funcs[feature, cls](feature, cls)
                    for feature, cls in self.fcs])

    def _compute_prob_y_given_x(self, _x, _y):
        normalisation_constant = sum([
            math.exp(sum([self.weights[_feature] *
                          self.feature_funcs[_feature, cls](_feature, cls)
                          for _feature in _x]))
            for cls in self.classes])

        return math.exp(sum([
            self.weights[_feature] *
            self.feature_funcs[_feature, _y](_feature, _y)
            for _feature in _x])) / normalisation_constant

    def _compute_estimated_expectation(self):
        return sum([self.empirical_fprob[feature] *
                    self._compute_prob_y_given_x(feature, cls) *
                    self.feature_funcs[feature, cls](feature, cls)
                    for feature, cls in self.fcs])

    def train(self, x_train, y_train, algorithm="IIS"):
        self.classes = set(y_train)
        self.train_set = list(zip(x_train, y_train))
        self.train_set_num = sum(1 for _ in self.train_set)  # len(train_set)
        # functions that extracts feature
        self.features = set([_feature for _x in x_train for _feature in _x])
        # a list of pairs of feature and class occurred in training set
        self.fcs = set(list(itertools.product(self.features,
                                              self.classes)))
        self.feature_funcs_example = dict((fc, generate_feature_func(*fc)
                                           for fc in self.fcs))
        self.feature_funcs = self.feature_funcs_example
        self.weights = dict().fromkeys(self.fcs)
        if algorithm == "IIS":
            self._train_max_ent_clf_with_iis()
        elif algorithm == "GIS":
            self._train_max_ent_clf_with_gis()
        elif algorithm == "BFGS":
            self._train_max_ent_clf_with_bfgs()

        # TODO: finish the return message of the max ent clf instance that is
        #       trained
        return "Max Entropy Classifier trained:\n" \
               "-max iteration: 200\n" \
               "-algorithm:{0}\n" \
               "" \
               "".format(algorithm)

    @staticmethod
    def convergence(new_weights, old_weights):
        for new_weight, old_weight in zip(new_weights, old_weights):
            if abs(new_weight - old_weight) >= EPSILON:
                return False
        return True

    def _train_max_ent_clf_with_gis(self):
        """ Generalized Iterative Scaling

        """
        self.empirical_ffreq = self._compute_empirical_ffreq()
        self.empirical_fprob = dict(
            (feature, count / self.train_set_num
             for feature, count in self.empirical_ffreq.items()))
        self.empirical_fcfreq = self._compute_empirical_fcfreq()
        self.empirical_fcprob = \
            dict((fc, fccount / self.train_set_num
                  for fc, fccount in self.empirical_fcfreq.items()))

        # the empirical expected value of f_i
        empirical_expectation = self._compute_empirical_expectation()
        # the expected value of f_i according to model
        estimated_expectation = self._compute_estimated_expectation()

        # In theory C can be any constant greater than or equal to the figure in
        #
        #         C = max(∑(f_i(x, y)))
        #                 x,y
        #
        # In practice C is maximised over the (x,y) pairs in the training data,
        # since 1/C determines the rate of convergence of the algorithm it is
        # preferable to keep C as small as possible.
        c = max(set([len(_x) for _x, _y in self.train_set]))

        for _ in range(self.max_iteration):
            for feature in self.features:
                self.tmp_weights = self.weights
                delta = math.log(
                    empirical_expectation / estimated_expectation) / c
                self.weights[feature] += delta
            if self.convergence(self.weights, self.tmp_weights):
                break

    # TODO: IIS Algorithm
    def _train_max_ent_clf_with_iis(self):
        """ Improved Iterative Scaling

        Steps:
            - Start with some (arbitrary) value for weight_feature
            - Repeat until convergence:
                          ∂B(g)
                -- Solve ------- = ∑ p(x, y)f_i(x, y) - ∑ p(x) ∑ p(y|x)f_i(x, y) e^w_i * f#(x, y)
                         ∂(d_i)   x,y                   x

                       d_i can be computed using NewTon-Raphson method:

                           d_i(k+1) = d_i(k) - g(d_i(k)) / g'(d_i(k))

                -- Set w_i <-- w_i + d_i

        """
        self.empirical_ffreq = self._compute_empirical_ffreq()
        self.empirical_fprob = dict(
            (feature, count / self.train_set_num
             for feature, count in self.empirical_ffreq.items()))
        self.empirical_fcfreq = self._compute_empirical_fcfreq()
        self.empirical_fcprob = \
            dict((fc, fccount / self.train_set_num
                  for fc, fccount in self.empirical_fcfreq.items()))

        def f_hash(x, y):
            return sum([feature_func(x, y) for feature_func in self.feature_funcs])

        goal = sum([self.empirical_fcprob[fc] * self.feature_funcs[fc] for fc in self.fcs]) - \
        sum([self.empirical_fprob[feature] * sum([self.empirical_fcprob[feature, cls] * self.feature_funcs[feature, cls] * math.e**(self.weights[feature]*f_hash(feature, cls))  for cls in self.classes]) for feature in self.features])
        for _ in range(self.max_iteration):
            tmp_weight = self.weights
            for feature in self.features:
                delta = newton_raphson_method(goal)
                self.weights[feature] += delta
            if self.convergence(self.weights, tmp_weight):
                break

    # TODO: BFGS Algorithm
    def _train_max_ent_clf_with_bfgs(self):
        pass

    def classify(self, x):
        prob = {}
        for cls in self.classes:
            prob[cls] = self._compute_prob_y_given_x(x, cls)
        return max(prob, key=prob.get)

    @classmethod
    def test(cls):
        max_ent_clf = cls()
        max_ent_clf.train()


if __name__ == "__main__":
    MaxEntropyClassifier.test()
