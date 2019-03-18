#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
from math import log2
import numpy as np
import pandas as pd
from Dataset import WaterMelonDataset, LoanApplierDataset


WATERMELON_DATASET = WaterMelonDataset()
LOAN_APPLIER_DATASET = LoanApplierDataset()


class DecisionTreeNode:
    def __init__(self, feature):
        self.feature = feature
        self.branches = {}

    def insert_node(self, val, node):
        self.branches[val] = node

    def predict(self, features):
        self.branches[features[self.feature]].predict(features)


class DecisionTree:
    def __init__(self):
        self.root_node = None

    def compute_empirical_entropy(self, y_train):
        return -sum(
                    sum(1 if y_type == y else 0 for y in y_train) / len(y_train)
                    * log2(
                           sum(1 if y_type == y else 0 for y in y_train)
                           / len(y_train)
                      )
                    for y_type in set(y_train)
                )

    def compute_empirical_conditional_entropy(self, feature_train, y_train):
        return sum(
                   sum(
                       1 if x == feature else 0 for x in feature_train
                   ) / len(y_train)
                   * self.compute_empirical_entropy(
                       [y for x, y in zip(feature_train, y_train)
                        if x == feature]
                     )
                   for feature in set(feature_train)
               )

    def compute_information_gain(self, A, D):
        return self.compute_empirical_entropy(D) - \
               self.compute_empirical_conditional_entropy(A, D)

    def compute_information_gain_ratio(self, A, D):
        return self.compute_information_gain(A, D) / \
               self.compute_empirical_entropy(D)

    def compute_gini_index(self):
        pass

    def _construct(self, train_data):
        pass

    def _prune(self):
        pass

    def fit(self, train_data: pd.DataFrame):
        self._construct(train_data)
        self._prune()

    def predict(self, x):
        self.root_node.predict(x)
