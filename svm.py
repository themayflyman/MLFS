#!/usr/bin/python
# -*- encoding: utf8 -*-

import numpy as np


class SupportVectorMachine:
    """Support Vector Machine.

    A implementation of Support Vector Machine with Sequential Minial
    Optimization algorithm.

    Primal formulation:
            N           N
    Min 1/2 ∑ w_i^2 + C ∑ §_i
           i=1         i=1
        s.t., y_i(w * x_i + b) ≥ 1 - §_i, i = 1,.....,N

    Dual formulation:
            N  N                                      N
    Min 1/2 ∑ ∑ ɑ_i * a_j * y_i * y_j * K(x_i, x_j) - ∑ ɑ_i
     ɑ    i=1 j=1                                    i=1
                              N
        s.t., 0 ≤ ɑ_i ≤ C and ∑ a_i * y_i = 0, i = 1,......,N
                             i=1

    Parameters
    ----------
    max_iteration : int (default=200)
        Maximum number of iterations for the algorithm.

    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    tol : float, optional (default=1e-3)
          Tolerance for stopping criteria.

    kernel : string, optional (default='gaussian')
        kernel function for the model

    Examples
    --------
    >>> from dataset import IrisDataset
    >>> from sklearn.model_selection import train_test_split
    >>> IRIS_DATASET = IrisDataset()
    >>> x_train, x_test, y_train, y_test = \
    ...        train_test_split(IRIS_DATASET.data[:100],
    ...                         IRIS_DATASET.target[:100])
    >>> svm = SupportVectorMachine(kernel='linear')
    >>> svm.train(x_train, y_train)
    SupportVectorMachine(kernel=linear, C=1.0, tol=0.001, max_iteration=200)
    >>> svm.classify(x_test[0])
    1
    >>> svm.score(x_test, y_test)
    0.96

    Notes
    -----

    References
    ----------
    李航 《统计学习方法》
    CS229 Simplified SMO algorithm (http://cs229.stanford.edu/materials/smo.pdf)
    John C. Platt Fast Training of Support Vector Machines using Sequential Minial Optimization (http://www.cs.utsa.edu/~bylander/cs6243/smo-book.pdf)
    """
    def __init__(self, **kwargs):
        self.max_iteration = kwargs.get("max_iteration", 200)
        # C is essentially a regularisation parameter, which controls the
        # trade-off between achieving a low error on the training data and
        # minimising the norm of the weights
        self.C = kwargs.get("C", 1.0)
        # numerical tolerance
        self.tol = kwargs.get('tol', 1e-3)
        self.kernel = kwargs.get("kernel", "gaussian")
        if self.kernel == "linear":
            self.kernel_func = self._linear_kernel_func
        elif self.kernel == "gaussian":
            self.kernel_func = self._gaussian_kernel_func
        elif self.kernel == "poly":
            self.kernel_func = self._polynomial_kernel_func
        else:
            self.kernel_func = None
            raise ValueError("Unsupported Kernel")

        self.degree = kwargs.get("degree", 3)

        self.sample_num = None

        self.weights = None
        # Threshold
        self.b = 0
        # Lagrange multipliers
        self.alpha = None

        self.prediction_error_cache = None

        self._x_train = None
        self._y_train = None

        self.decision_func = None

    @staticmethod
    def _linear_kernel_func(x, z):
        return np.dot(x, z.T)

    def _polynomial_kernel_func(self, x, z):
        return pow((np.dot(x, z) + 1), self.degree)

    @staticmethod
    def _gaussian_kernel_func(x, z, sigma=1):
        return np.exp(-pow(np.linalg.norm(x - z), 2) / 2 * pow(sigma, 2))

    def if_violate_kkt_conditions(self, index):
        """Check if a certain sample of given index violates KKT conditions
        For this problem, the KKT conditions are as following:

        g(x) = w.T * x + b

        a_i = 0         =>     y_i * g(x_i) >= 1
        0 < a_i < C     =>     y_i * g(x_i) = 1
        a_i = C         =>     y_i * g(x_i) <= 1

        Parameters
        ----------
        index: int

        Returns
        -------
        v: boolean
            if a given sample violates KKT conditions
        """
        r = self.prediction_error_cache[index] * self._y_train[index]
        if (self.alpha[index] < self.C and r < -self.tol) or \
                (self.alpha[index] > 0 and r > self.tol):
            return True
        else:
            return False

    def _primal_hypothesis_func(self, x):
        return np.sign(np.dot(self.weights, x) + self.b)

    def _dual_hypothesis_func(self, x):
        return sum([alpha * _y * self.kernel_func(_x, x)
                    for alpha, _x, _y in zip(self.alpha,
                                             self._x_train,
                                             self._y_train)]
                   ) + self.b

    def _compute_prediction_error(self, x, y):
        return sum([_alpha * _y * self.kernel_func(_x, x)
                    for _alpha, _x, _y in zip(self.alpha,
                                              self._x_train,
                                              self._y_train)]
                   ) + self.b - y

    def sequential_minial_optimize(self):
        # A cached error value for every non-bound example in the training
        # set and within the inner loop it chooses an error to approximately
        # maximize the step size.
        self.prediction_error_cache = [self._compute_prediction_error(_x,
                                                                      _y)
                                       for _x, _y in zip(self._x_train,
                                                         self._y_train)]

        iteration = 0
        while iteration < self.max_iteration:
            num_updated_alphas = 0
            for j in range(self.sample_num):
                if self.if_violate_kkt_conditions(j):
                    # Given the first ɑ_i, the inner loop looks for a
                    # non-boundary that maximizes |E_2 - E_1|. If this does not
                    # make progress, it starts a sequential scan through the
                    # non-boundary examples, starting at a random position; if
                    # this fails too, it starts a sequential scan through all
                    # examples, also starting at a random position.
                    if self.prediction_error_cache[j] > 0:
                        i = int(np.argmin(self.prediction_error_cache))
                    else:
                        i = int(np.argmax(self.prediction_error_cache))
                    alpha_i = self.alpha[i]
                    error_i = self.prediction_error_cache[i]
                    x_i = self._x_train[i]
                    y_i = self._y_train[i]
                    alpha_j = self.alpha[j]
                    error_j = self.prediction_error_cache[j]
                    x_j = self._x_train[j]
                    y_j = self._y_train[j]
                    if y_i != y_j:
                        lower_bound = max(0, alpha_i - alpha_j)
                        upper_bound = min(self.C, self.C+alpha_j - alpha_i)
                    else:
                        lower_bound = max(0, alpha_i + alpha_j - self.C)
                        upper_bound = min(self.C, alpha_i + alpha_j)

                    eta = \
                        self.kernel_func(x_i, x_i) + \
                        self.kernel_func(x_j, x_j) - \
                        2 * self.kernel_func(x_i, x_j)

                    if eta > 0:
                        unclipped_new_alpha_j = \
                            alpha_j + y_j * (error_i - error_j) / eta

                        if unclipped_new_alpha_j > upper_bound:
                            new_alpha_j = upper_bound
                        elif unclipped_new_alpha_j < lower_bound:
                            new_alpha_j = lower_bound
                        else:
                            new_alpha_j = unclipped_new_alpha_j
                    else:
                        fi = y_i * (error_i + self.b) - \
                             alpha_i * self.kernel_func(x_i, x_i) - \
                             y_i * y_j * alpha_j * self.kernel_func(x_i, x_j)
                        fj = \
                            y_j * (error_j + self.b) - \
                            y_i * y_j * alpha_i * self.kernel_func(x_i, x_j) - \
                            alpha_j * self.kernel_func(x_i, x_j)
                        lower_bound_i = alpha_i + y_i * y_j * (alpha_j -
                                                               lower_bound)
                        upper_bound_i = alpha_i + y_i * y_j * (alpha_j -
                                                               upper_bound)
                        lower_bound_obj = \
                            lower_bound_i * fi + \
                            lower_bound * fj + \
                            0.5 * lower_bound_i**2 * self.kernel_func(x_i,
                                                                      x_i) + \
                            0.5 * lower_bound**2 * self.kernel_func(x_j, x_j) +\
                            y_i * y_j * lower_bound * lower_bound_i * \
                            self.kernel_func(x_i, x_j)
                        upper_bound_obj = \
                            upper_bound_i * fi + \
                            upper_bound * fj + \
                            0.5 * upper_bound_i**2 * self.kernel_func(x_i,
                                                                      x_i) + \
                            0.5 * upper_bound**2 * self.kernel_func(x_j, x_j) +\
                            y_i * y_j * upper_bound * upper_bound_i * \
                            self.kernel_func(x_i, x_j)

                        if lower_bound_obj < upper_bound_obj - 1e-3:
                            new_alpha_j = lower_bound
                        elif lower_bound_obj > upper_bound_obj + 1e-3:
                            new_alpha_j = upper_bound
                        else:
                            new_alpha_j = alpha_j

                    if new_alpha_j < 1e-8:
                        new_alpha_j = 0
                    elif new_alpha_j > (self.C - 1e-8):
                        new_alpha_j = self.C

                    # if alphas can't be optimized within epsilon,
                    # skip this pair
                    if abs(new_alpha_j - alpha_j) < \
                            1e-3 * (alpha_j + new_alpha_j + 1e-3):
                        continue

                    new_alpha_i = alpha_i + y_i * y_j * (alpha_j - new_alpha_j)

                    b_i = \
                        - error_i - \
                        y_i * (new_alpha_i - alpha_i) * self.kernel_func(x_i,
                                                                         x_i) -\
                        y_j * (new_alpha_j - alpha_j) * self.kernel_func(x_i,
                                                                         x_j) +\
                        self.b
                    b_j = \
                        - error_j - \
                        y_i * (new_alpha_i - alpha_i) * self.kernel_func(x_i,
                                                                         x_j) -\
                        y_j * (new_alpha_j - alpha_j) * self.kernel_func(x_j,
                                                                         x_j) +\
                        self.b

                    if 0 < alpha_i < self.C:
                        new_b = b_i
                    elif 0 < alpha_j < self.C:
                        new_b = b_j
                    else:
                        new_b = (b_i + b_j) * 0.5

                    # update alpha, b and error cache
                    if self.kernel == "linear":
                        self.weights += \
                            y_i * (new_alpha_i - self.alpha[i]) * x_i + \
                            y_j * (new_alpha_j - self.alpha[j]) * x_j
                    self.alpha[i] = new_alpha_i
                    self.alpha[j] = new_alpha_j
                    self.b = new_b
                    self.prediction_error_cache[i] = \
                        self._compute_prediction_error(x_i, y_i)
                    self.prediction_error_cache[j] = \
                        self._compute_prediction_error(x_j, y_j)

                    num_updated_alphas += 1

            if num_updated_alphas == 0:
                iteration += 1
            else:
                iteration = 0

    def train(self, x_train, y_train):
        self.weights = np.zeros(x_train.shape[1])
        self.sample_num = len(x_train)
        self.alpha = np.zeros(self.sample_num)
        self._x_train = x_train
        self._y_train = y_train

        self.sequential_minial_optimize()

        if self.kernel == "linear":
            self.decision_func = self._primal_hypothesis_func
        else:
            self.decision_func = self._dual_hypothesis_func

        return "SupportVectorMachine(" \
               "kernel={0}, " \
               "C={1}, tol={2}, " \
               "max_iteration={3})".format(self.kernel,
                                           self.C,
                                           self.tol,
                                           self.max_iteration)

    def classify(self, x):
        prediction = self.decision_func(x)
        return 1 if prediction > 0 else 0

    def score(self, x_test, y_test):
        return sum([1 if self.classify(_x) == _y else 0
                    for _x, _y in zip(x_test, y_test)]) / len(x_test)

    @classmethod
    def test(cls):
        from dataset import IrisDataset
        from sklearn.model_selection import train_test_split

        iris_dataset = IrisDataset()
        x_train, x_test, y_train, y_test = \
            train_test_split(iris_dataset.data[:100],
                             iris_dataset.target[:100])

        svm = cls(kernel="linear", max_iteration=200)
        print(svm.train(x_train, y_train))
        print(svm.score(x_test, y_test))


if __name__ == "__main__":
    SupportVectorMachine.test()
