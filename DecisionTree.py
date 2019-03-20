#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
from math import log2
from collections import Counter
import numpy as np
import pandas as pd
from Dataset import WaterMelonDataset, LoanApplierDataset


WATERMELON_DATASET = WaterMelonDataset()
LOAN_APPLIER_DATASET = LoanApplierDataset()


class DecisionTreeNode:
    def __init__(self, feature_label=None, cls_label=None):
        self._feature_label = feature_label
        self._cls_label = cls_label
        self.nodes = {}

    @property
    def feature_label(self):
        return self._feature_label

    @feature_label.setter
    def feature_label(self, feature_label):
        self._feature_label = feature_label

    @property
    def cls_label(self):
        return self._cls_label

    @cls_label.setter
    def cls_label(self, cls_label):
        self._cls_label = cls_label

    def insert_node(self, feature_value, node):
        pass

    def remove_node(self, node):
        pass

    def decide(self, x):
        pass


class DecisionTree:
    def __init__(self, epsilon=0.01, criterion="information gain"):
        self.epsilon = epsilon
        self.criterion = criterion
        self.root_node = DecisionTreeNode()

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

    def _construct(self, decision_tree_node, x_train, y_train, feature_labels):
        classes = set(y_train)
        # If all cases in dataset belong to a single class
        if len(classes) == 1:
            decision_tree_node.cls_label = classes.pop()
        # If there's no more feature left
        elif not feature_labels:
            cls_of_max_instances = max(Counter(y_train))
            decision_tree_node.cls_label = cls_of_max_instances
        else:
            information_gain = dict.fromkeys(feature_labels)
            for feature_label in feature_labels:
                information_gain[feature_label] = \
                    self.compute_information_gain(x_train[:, feature_label],
                                                  y_train)
            feature_label_with_max_information_gain = max(information_gain)
            if information_gain[feature_label_with_max_information_gain] < \
                    self.epsilon:
                cls_of_max_instances = max(Counter(y_train))
                decision_tree_node.cls_label = cls_of_max_instances
            else:
                feature_labels.remove(feature_label_with_max_information_gain)
                feature_values = \
                    set(x_train[:, feature_label_with_max_information_gain])
                for feature_value in feature_values:
                    next_decision_tree_node = DecisionTreeNode()
                    decision_tree_node.insert_node(
                        feature_value,
                        self._construct(
                            next_decision_tree_node,
                            x_train,
                            y_train,
                            feature_labels
                        )
                    )

        return decision_tree_node

    def _prune(self):
        pass

    def fit(self, x_train, y_train):

        self._construct(decision_tree_node=self.root_node,
                        x_train=x_train,
                        y_train=y_train,
                        feature_labels=list(range(x_train.shape[1])))
        self._prune()

    def predict(self, x):
        pass


d = DecisionTree()
d.fit(LOAN_APPLIER_DATASET.data, LOAN_APPLIER_DATASET.target)
print(d.root_node.nodes)
