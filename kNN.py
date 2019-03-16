#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
import heapq
from typing import List
from sklearn.datasets import load_iris

# TODO: Use Mixin


class LpDistance(float):
    def __new__(cls, x_i, x_j, p):
        tmp_sum = 0
        if len(x_i) == len(x_j) and len(x_i) > 1:
            for i, j in zip(x_i, x_j):
                tmp_sum += math.pow(abs(i - j), p)
            lp_distance = math.pow(tmp_sum, 1/p)
            # HACK
            # lp_distance = math.pow(sum(math.pow(abs(i - j), p)
            #                        for i, j in zip(x_i, x_j)), 1/p)
            return super().__new__(cls, lp_distance)
        else:
            raise ValueError("The dimension of x_i and x_j must be the same")

    def __init__(self, x_i, x_j, p):
        float.__init__(self)
        self.x_i = x_i
        self.x_j = x_j
        self.p = p


class EuclideanDistance(LpDistance):
    def __new__(cls, x_i, x_j):
        return super().__new__(cls, x_i, x_j, 2)

    def __init__(self, x_i, x_j):
        LpDistance.__init__(self, x_i, x_j, 2)
        self.x_i = x_i
        self.x_j = x_j


class ManhattanDistance(LpDistance):
    def __new__(cls, x_i, x_j):
        return super().__new__(cls, x_i, x_j, 1)

    def __init__(self, x_i, x_j):
        LpDistance.__init__(self, x_i, x_j, 1)
        self.x_i = x_i
        self.x_j = x_j


class kdTreeNode:
    def __init__(self, split_point=None, depth=0,
                 left_child=None, right_child=None):
        self.split_point = split_point
        self.depth = depth
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        return "kdTreeNode(split_point={}, depth={})" \
            .format(self.split_point, self.depth)


# k in kTree represents the number of dimensions,
# which is different from k in kNN
class kdTree:
    def __init__(self, points=None):
        def select_axis(depth, k):
            return depth % k

        def find_median(pts, axis):
            pts.sort(key=lambda k: k[0][axis])
            median = len(pts) // 2
            return pts, median

        def construct(pts, depth=0):
            if not pts:
                return None
            k = len(pts[0][0])
            axis = select_axis(depth, k)

            pts, median = find_median(pts, axis)

            return kdTreeNode(split_point=pts[median],
                              depth=depth,
                              left_child=construct(pts[:median], depth + 1),
                              right_child=construct(pts[median+1:], depth + 1))
        if points:
            self.root_node = construct(points)

    def pre_order_traversal(self):
        self._pre_order_traversal(self.root_node)

    def _pre_order_traversal(self, node):
        if node:
            print(node)
            self._pre_order_traversal(node.left_child)
            self._pre_order_traversal(node.right_child)

    @classmethod
    def test(cls):
        test_points = [((2, 3), 1), ((5, 4), 1), ((9, 6), 1),
                       ((4, 7), 1), ((8, 1), 0), ((7, 2), 0)]
        kd_tree = cls(test_points)
        kd_tree.pre_order_traversal()

    # TODO: plot kd tree as well as the splitting on feature space
    def plot(self):
        pass


class NearerPoint:
    def __init__(self, point, distance_2_search_point=None):
        self.point = point
        self._distance_2_search_point = distance_2_search_point

    @property
    def distance_to_search_point(self):
        return self._distance_2_search_point

    @distance_to_search_point.setter
    def distance_2_search_point(self, distance):
        self._distance_2_search_point = distance

    def compute_distance_2_search_point(self, search_point, distance_cls):
        self._distance_2_search_point = distance_cls(search_point, self.point[0])

    def __lt__(self, other):
        return self.distance_2_search_point > \
                    other.distance_2_search_point

    def __eq__(self, other):
        return self.distance_2_search_point == \
               other.distance_2_search_point

    def __repr__(self):
        return "Point: {}".format(self.point[0])


class MaxHeap4NearerPoint:
    def __init__(self, iterable: List[NearerPoint] = None, size=float('inf')):
        self.heap = []
        self.size = size
        if iterable:
            for item in iterable:
                self.push(item)
        self._nearest_point_so_far = None

    def push(self, item: NearerPoint):
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, item)
        else:
            heapq.heappushpop(self.heap, item)

    def pop(self):
        return heapq.heappop(self.heap)

    @property
    def nearest_point_so_far(self):
        # Since we reverse __lt__ in NearerPoint so we call heapq.nlargest()
        # when we want the nearest point, this can also be implemented by
        # checking all nodes in the max heap
        return heapq.nlargest(1, self.heap)[0]

    def __iter__(self):
        return iter(self.heap)

    def __getitem__(self, index):
        return self.heap[index]

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return str(self.heap)


class kNN:
    def __init__(self, distance_cls=EuclideanDistance):
        self.distance_cls = distance_cls
        self.dimension = 0
        self._k = 1
        self.kd_tree = kdTree()
        # an array to store k nearest points and their distance to the search
        # point
        self.k_nearest = MaxHeap4NearerPoint(size=self.k)

    # k is the number of nearest points we want to search
    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        if k < 0:
            raise ValueError("k must be greater than 0")
        else:
            self._k = k
            self.k_nearest = MaxHeap4NearerPoint(size=self.k)

    def fit(self, x_train, y_train):
        self.dimension = len(x_train[0])
        self.kd_tree = kdTree([(x, y) for x, y in zip(x_train, y_train)])

    def search(self, kd_tree_node, x):
        # 1. Starting with the root node, it moves down the tree, recursively,
        # in the same way that it would if the search point were being inserted.
        if kd_tree_node:
            axis = kd_tree_node.depth % self.dimension
            if x[axis] - kd_tree_node.split_point[0][axis] < 0:
                self.search(kd_tree_node.left_child, x)
            else:
                self.search(kd_tree_node.right_child, x)

            distance = self.distance_cls(x, kd_tree_node.split_point[0])
            self.k_nearest.push(NearerPoint(kd_tree_node.split_point, distance))

            farthest_point_so_far = self.k_nearest.nearest_point_so_far
            print(farthest_point_so_far)
            # checks whether there could be any points on the other side of
            # the splitting plane that are closer to the search point than the
            # current best.
            # It's done by intersecting the hypersphere around the search point
            # x that has a radius equal to the current nearest distance,
            # implemented as a simple comparison (as following) to see whether
            # the distance between the splitting coordinate of the search point
            # and current node is lesser than the distance from the search
            # point to the current best.

            # distance between the splitting coordinate of the search point
            # and current node
            distance_in_axis = abs(x[axis] - kd_tree_node.split_point[0][axis])
            if farthest_point_so_far.distance_2_search_point > distance_in_axis:
                # If the hyper crosses the plane, there could be nearer points
                # on the other side of the plane, so the algorithm must move
                # down the other branch of the tree from current node looking
                # for closer points, following the same recursive process as
                # the entire search. Otherwise it continues walking up the
                # tree, and the entire branch on the other side of that node
                # is eliminated.
                if x[axis] - kd_tree_node.split_point[0][axis] < 0:
                    self.search(kd_tree_node.right_child, x)
                else:
                    self.search(kd_tree_node.left_child, x)

    def predict(self, x):
        self.search(self.kd_tree.root_node, x)
        class_of_all_nearest_points = [nearest_point.point[1]
                                       for nearest_point in self.k_nearest]
        return max(set(class_of_all_nearest_points),
                   key=class_of_all_nearest_points.count)

    @classmethod
    def test(cls):
        test_points = [((2, 3), 1), ((5, 4), 1), ((9, 6), 1),
                       ((4, 7), 1), ((8, 1), 0), ((7, 2), 0)]
        knn_clf = cls()
        knn_clf.k = 3
        knn_clf.fit([t[0] for t in test_points], [t[1] for t in test_points])
        knn_clf.predict((3, 4.5))
        assert knn_clf.k_nearest.nearest_point_so_far.point[0] == (2, 3)


if __name__ == "__main__":
    kdTree.test()
    kNN.test()

