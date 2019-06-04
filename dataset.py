#!/usr/bin/python
# -*- encoding: utf8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


IRIS = load_iris()


class Dataset(metaclass=ABCMeta):
    def __init__(self):
        self._dataset = pd.DataFrame()
        self._columns = []
        self._feature_names = []
        self._init_feature_names()
        self._init_columns()
        self._init_dataset()
        self.data = np.array(self._dataset)[:, :-1]
        self.target = np.array(self._dataset)[:, -1]

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        raise RuntimeError("To update dataset, call feed(), delete()")

    @abstractmethod
    def _init_dataset(self):
        pass

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        if self._dataset[0]:
            if len(columns) != len(self._dataset[0]) - 1:
                raise ValueError("wrong length of the columns")
        self._columns = columns

    def _init_columns(self):
        self._columns = self.feature_names + ['class']

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        if self._dataset[0]:
            if len(feature_names) != len(self._dataset[0]) - 1:
                # TODO: polish the error message
                raise ValueError("wrong length of the columns")
        self._feature_names = feature_names

    @abstractmethod
    def _init_feature_names(self):
        pass

    def feed(self, data):
        self._dataset.append(data)

    def delete(self, data):
        self._dataset.remove(data)

    # TODO: train_test_split function
    def train_test_split(self):
        pass


class IrisDataset(Dataset):
    def _init_dataset(self):
        self._dataset = \
            pd.DataFrame([data.tolist() + [target]
                          for data, target in zip(IRIS.data, IRIS.target)],
                         columns=self.columns)

    def _init_feature_names(self):
        self._feature_names = IRIS.feature_names


class WaterMelonDataset(Dataset):
    def _init_dataset(self):
        self._dataset = pd.DataFrame([
            ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
            ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
            ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
            ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
            ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
        ], columns=self.columns)

    def _init_feature_names(self):
        self._feature_names = \
            ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']


class LoanApplierDataset(Dataset):
    def _init_dataset(self):
        self._dataset = pd.DataFrame([
            ['青年', '否', '否', '一般', '否'],
            ['青年', '否', '否', '好', '否'],
            ['青年', '是', '否', '好', '是'],
            ['青年', '是', '是', '一般', '是'],
            ['青年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '好', '否'],
            ['中年', '是', '是', '好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '好', '是'],
            ['老年', '是', '否', '好', '是'],
            ['老年', '是', '否', '非常好', '是'],
            ['老年', '否', '否', '一般', '否'],
            ], columns=self.columns)

    def _init_feature_names(self):
        self._feature_names = ['年龄', '有工作', '有自己的房子', '信贷情况']
