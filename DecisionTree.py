#!/usr/bin/python
# -*- encoding: utf8 -*-

from math import log2
from collections import Counter
import numpy as np
from Dataset import WaterMelonDataset, LoanApplierDataset


WATERMELON_DATASET = WaterMelonDataset()
LOAN_APPLIER_DATASET = LoanApplierDataset()


class DecisionTreeNode:
    def __init__(self, x_train=None, y_train=None,
                 feature_label=None, cls_label=None):
        self._feature_label = feature_label
        self._cls_label = cls_label
        self.x_train = x_train
        self.y_train = y_train
        self.child_nodes = {}

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

    def insert_child_node(self, feature_value, node):
        self.child_nodes[feature_value] = node

    @classmethod
    def prune_transform(cls, decision_tree_node):
        # prune
        decision_tree_node.child_nodes = {}
        # transform the node into a leaf node
        cls_of_max_instances = max(Counter(decision_tree_node.y_train))
        decision_tree_node.cls_label = cls_of_max_instances
        decision_tree_node.feature_label = None

    def decide(self, x):
        return self.cls_label if self.cls_label else \
            self.child_nodes[x[self.feature_label]].decide(x)


class DecisionTree:
    def __init__(self, alpha=0.1, epsilon=0.01, criterion="information gain",
                 if_prune=True):
        self.alpha = alpha
        self.epsilon = epsilon
        self.criterion = criterion
        self.if_prune = if_prune
        self.root_node = None
        self._leaf_nodes = []

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

    # Information gain measures how much "information" a feature gives us about
    # the class.
    def compute_information_gain(self, A, D):
        return self.compute_empirical_entropy(D) - \
               self.compute_empirical_conditional_entropy(A, D)

    def compute_information_gain_ratio(self, A, D):
        return self.compute_information_gain(A, D) / \
               self.compute_empirical_entropy(D)

    def compute_criterion(self, x_train, y_train):
        if self.criterion == "information gain":  # ID3 Algorithm
            return self.compute_information_gain(x_train, y_train)
        elif self.criterion == "information gain ratio":  # C4.5 Algorithm
            return self.compute_information_gain_ratio(x_train, y_train)

    def cost_function(self, leaf_nodes):
        return sum(len(leaf_node.y_train) *
                   self.compute_empirical_entropy(leaf_node.y_train)
                   for leaf_node in leaf_nodes) + \
            self.alpha * len(leaf_nodes)

    def _construct(self, decision_tree_node, x_train, y_train, feature_labels):
        classes = set(y_train)
        # If all samples in dataset belong to a single class
        if len(classes) == 1:
            decision_tree_node.cls_label = classes.pop()
        # If there's no more feature left to construct the tree
        elif not feature_labels:
            cls_of_max_instances = max(Counter(y_train))
            decision_tree_node.cls_label = cls_of_max_instances
        else:
            criterion = dict.fromkeys(feature_labels)
            for feature_label in feature_labels:
                criterion[feature_label] = \
                    self.compute_criterion(x_train[:, feature_label], y_train)
            feature_label_with_max_criterion = max(criterion, key=criterion.get)
            if criterion[feature_label_with_max_criterion] < self.epsilon:
                cls_of_max_instances = max(Counter(y_train))
                decision_tree_node.cls_label = cls_of_max_instances
            else:
                feature_labels.remove(feature_label_with_max_criterion)
                decision_tree_node.feature_label = \
                    feature_label_with_max_criterion
                feature_values = set(x_train[:, feature_label_with_max_criterion])
                for feature_value in feature_values:
                    train_data = np.column_stack([x_train, y_train])
                    x_split = train_data[train_data[:, feature_label_with_max_criterion] == feature_value][:, :-1]
                    y_split = train_data[train_data[:, feature_label_with_max_criterion] == feature_value][:, -1]
                    child_decision_tree_node = \
                        DecisionTreeNode(x_train=x_split, y_train=y_split)
                    decision_tree_node.insert_child_node(
                        feature_value,
                        self._construct(
                            child_decision_tree_node,
                            x_split,
                            y_split,
                            feature_labels
                        )
                    )

        return decision_tree_node

    @property
    def leaf_nodes(self):
        return self.find_leaf_nodes()

    def find_leaf_nodes(self):
        leaf_nodes = []
        self._find_leaf_nodes(self.root_node, leaf_nodes)
        return leaf_nodes

    def _find_leaf_nodes(self, decision_tree, leaf_nodes):
        for child_node in decision_tree.child_nodes:
            if child_node.child_nodes:
                leaf_nodes += self._find_leaf_nodes(child_node, leaf_nodes)
            else:
                self._leaf_nodes += child_node.values()

    def _post_prune(self, decision_tree_node):
        # HACK: Dynamic programming
        if decision_tree_node.child_nodes:
            for child_node in decision_tree_node.child_nodes.values():
                self._post_prune(child_node)
        # Find all leaf nodes before pruning
        leaf_nodes_before_pruning = self.find_leaf_nodes()
        # Assume we pruned the branches, the leaf nodes would be left
        leaf_nodes_after_pruning = [
            leaf_node for leaf_node in self._leaf_nodes
            if leaf_node not in decision_tree_node.child_nodes.values()
        ] + [decision_tree_node]
        # Compute the cost before and after the pruning
        cost_before_pruning = self.cost_function(leaf_nodes_before_pruning)
        cost_after_pruning = self.cost_function(leaf_nodes_after_pruning)
        if cost_after_pruning < cost_before_pruning:
            # prune if the cost is less than the cost after the pruning
            DecisionTreeNode.prune_transform(decision_tree_node)

    def post_prune(self):
        self._post_prune(self.root_node)

    def fit(self, x_train, y_train):
        self.root_node = DecisionTreeNode(x_train=x_train, y_train=y_train)
        self._construct(decision_tree_node=self.root_node,
                        x_train=x_train,
                        y_train=y_train,
                        feature_labels=list(range(x_train.shape[1])))
        self._find_leaf_nodes(self.root_node)
        if self.if_prune:
            self.post_prune()

    def predict(self, x):
        return self.root_node.decide(x)

    @classmethod
    def test(cls):
        decision_tree = cls()
        decision_tree.fit(LOAN_APPLIER_DATASET.data,
                          LOAN_APPLIER_DATASET.target)

