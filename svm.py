#!/usr/bin/python
# -*- encoding: utf8 -*-

from random import randint
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import IrisDataset


IRIS_DATASET = IrisDataset()


class SupportVectorMachine:
    """A implementation of SVM (support Vector Machine)

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

    """
    def __init__(self, **kwargs):
        self._max_iteration = kwargs.get("max_iteration", 200)
        # C is essentially a regularisation parameter, which controls the
        # trade-off between achieving a low error on the training data and
        # minimising the norm of the weights
        self._C = kwargs.get("C", 1)
        self._tol = kwargs.get('tol', 1e-3)
        self.kernel = kwargs.get("kernel", "gaussian")
        if self.kernel == "linear":
            self.kernel_func = self._linear_kernel_func
        elif self.kernel == "gaussian":
            self.kernel_func = self._gaussian_kernel_func
        elif self.kernel == "poly":
            self.kernel_func = self._polynomial_kernel_func
        elif self.kernel == "string":
            self.kernel_func = self._string_kernel_func
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

    @property
    def max_iteration(self):
        return self._max_iteration

    @max_iteration.setter
    def max_iteration(self, val):
        self._max_iteration = val

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, val):
        self._C = val

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, val):
        self._tol = val

    @staticmethod
    def _linear_kernel_func(x, z):
        return np.dot(x, z.T)

    def _polynomial_kernel_func(self, x, z):
        return pow((np.dot(x, z) + 1), self.degree)

    @staticmethod
    def _gaussian_kernel_func(x, z, sigma):
        return np.exp(-pow(np.linalg.norm(x - z), 2) / 2 * pow(sigma, 2))

    # TODO: finish string kernel func
    @staticmethod
    def _string_kernel_func():
        pass

    def if_violate_kkt_conditions(self, index):
        R = self.prediction_error_cache[index] * self._y_train[index]
        if (self.alpha[index] < self.C and R < -self.tol) or \
                (self.alpha[index] > 0 and R > self.tol):
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

    def _outer_loop(self):
        # The outer loop alternates between one sweep through all examples and
        # as many sweeps as possible through the non-boundary examples
        # (those with 0 < ɑ_i < C), selecting the example that violates KKT
        # condition.
        index_of_samples_violate_kkt_conditions = []
        index_of_non_bound_sample = [i for i in range(len(self.alpha))
                                     if 0 < self.alpha[i] < self.C]

        for nbs_index in index_of_non_bound_sample:
            if self.if_violate_kkt_conditions(nbs_index):
                index_of_samples_violate_kkt_conditions.append(nbs_index)

        # If all the support vectors on the boundary obey the KKT conditions,
        # loop through the whole training set to see if there's any sample
        # violates KKT conditions
        if not index_of_samples_violate_kkt_conditions:
            index_to_examine = [i for i in range(self.sample_num)
                                if i not in index_of_non_bound_sample]
            for index_te in index_to_examine:
                if self.if_violate_kkt_conditions(index_te):
                    index_of_samples_violate_kkt_conditions.append(index_te)

        if not index_of_samples_violate_kkt_conditions:
            return None

        random_index = randint(0,
                               len(index_of_samples_violate_kkt_conditions)-1)
        return index_of_samples_violate_kkt_conditions[0]

    def _inner_loop(self, index_i):
        # Given the first ɑ_i, the inner loop looks for a non-boundary that
        # maximizes |E_2 - E_1|. If this does not make progress, it starts a
        # sequential scan through the non-boundary examples, starting at a
        # random position; if this fails too, it starts a sequential scan
        # through all examples, also starting at a random position.
        if self.prediction_error_cache[index_i] > 0:
            index_j = np.argmin(self.prediction_error_cache)
        else:
            index_j = np.argmax(self.prediction_error_cache)

        return int(index_j)

    def sequential_minial_optimize(self):
        # A cached error value for every non-bound example in the training set
        # and within the inner loop it chooses an error to approximately
        # maximize the step size.
        self.prediction_error_cache = [self._compute_prediction_error(_x, _y)
                                       for _x, _y in zip(self._x_train,
                                                         self._y_train)]
        print(self.prediction_error_cache)
        for _ in range(self.max_iteration):
            # SMO users heuristics to choose which two Lagrange multipliers to
            # jointly optimize
            # The outer loop selects the first ɑ_i
            i = self._outer_loop()
            # If no sample violates KKT, the optimization is done
            if i is None:
                print("No sample violates KKT conditions, "
                      "current iteration: {}, "
                      "exiting SMO...".format(_))
                break
            # The inner loop selects the second ɑ_i that maximizes |E_2 - E_1|
            j = self._inner_loop(i)
            alpha_i = self.alpha[i]
            error_i = self.prediction_error_cache[i]
            x_i = self._x_train[i]
            y_i = self._y_train[i]
            if y_i == 0:
                print("HHHHHHHHHHHHHHHHHHHHH")
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
                self.kernel_func(x_i, x_i) + self.kernel_func(x_j, x_j) - 2 * \
                self.kernel_func(x_i, x_j)

            if eta > 0:
                unclipped_new_alpha_j = alpha_j + y_j * (error_i - error_j) / eta

                if unclipped_new_alpha_j > upper_bound:
                    new_alpha_j = upper_bound
                elif unclipped_new_alpha_j < lower_bound:
                    new_alpha_j = lower_bound
                else:
                    new_alpha_j = unclipped_new_alpha_j
            else:
                fi = y_i * (error_i + self.b) - alpha_i * self.kernel_func(x_i, x_i) - y_i * y_j * alpha_j * self.kernel_func(x_i, x_j)
                fj = y_j * (error_j + self.b) - y_i * y_j * alpha_i * self.kernel_func(x_i, x_j) - alpha_j * self.kernel_func(x_i, x_j)
                lower_bound_i = alpha_i + y_i * y_j * (alpha_j - lower_bound)
                upper_bound_i = alpha_i + y_i * y_j * (alpha_j - upper_bound)
                lower_bound_obj = lower_bound_i * fi + lower_bound * fj + 0.5 * lower_bound_i**2 * self.kernel_func(x_i, x_i) + 0.5 * lower_bound**2 * self.kernel_func(x_j, x_j) + y_i * y_j * lower_bound * lower_bound_i * self.kernel_func(x_i, x_j)
                upper_bound_obj = upper_bound_i * fi + upper_bound * fj + 0.5 * upper_bound_i**2 * self.kernel_func(x_i, x_i) + 0.5 * upper_bound**2 * self.kernel_func(x_j, x_j) + y_i * y_j * upper_bound * upper_bound_i * self.kernel_func(x_i, x_j)
                if lower_bound_obj < upper_bound_obj - 1e-3:
                    new_alpha_j = lower_bound
                elif lower_bound_obj > upper_bound_obj + 1e-3:
                    new_alpha_j = upper_bound
                else:
                    new_alpha_j = alpha_j

            # if abs(new_alpha_j - alpha_j) < 1e-3 * (alpha_j + new_alpha_j + 1e-3):
            #     continue

            new_alpha_i = alpha_i + y_i * y_j * (alpha_j - new_alpha_j)

            b_i = \
                error_i + \
                y_i * (new_alpha_i - alpha_i) * self.kernel_func(x_i, x_i) + \
                y_j * (new_alpha_j - alpha_j) * self.kernel_func(x_i, x_j) + \
                self.b
            b_j = \
                error_j + \
                y_i * (new_alpha_i - alpha_i) * self.kernel_func(x_i, x_j) + \
                y_j * (new_alpha_j - alpha_j) * self.kernel_func(x_j, x_j) + \
                self.b

            if 0 < alpha_i < self.C:
                new_b = b_i
            elif 0 < alpha_j < self.C:
                new_b = b_j
            else:
                new_b = (b_i + b_j) * 0.5

            # Update alpha, b and error cache
            if self.kernel == "linear":
                self.weights += \
                    y_i * (new_alpha_i - self.alpha[i]) * x_i + \
                    y_j * (new_alpha_j - self.alpha[j]) * x_j
            self.alpha[i] = new_alpha_i
            self.alpha[j] = new_alpha_j
            self.b = new_b
            self.prediction_error_cache[i] = self._compute_prediction_error(x_i,
                                                                            y_i)
            self.prediction_error_cache[j] = self._compute_prediction_error(x_j,
                                                                            y_j)

    def train(self, x_train, y_train):
        self.weights = np.zeros(x_train.shape[1])
        self.sample_num = len(x_train)
        self.alpha = np.zeros(self.sample_num)
        self._x_train = x_train
        self._y_train = y_train

        self.sequential_minial_optimize()

        if self.kernel == "linear":
            self.decision_func = self._linear_decision_func
        else:
            self.decision_func = self._nonlinear_decision_func

        # TODO: Message after the SVM is trained
        return "Trained message and the current status of svm"

    def _linear_decision_func(self, x):
        return self._primal_hypothesis_func(x)

    def _nonlinear_decision_func(self, x):
        return self._dual_hypothesis_func(x)

    def predict(self, x):
        prediction = self.decision_func(x)
        return 1 if prediction > 0 else 0

    def score(self, x_test, y_test):
        return sum([1 if self.predict(_x) == _y else 0
                    for _x, _y in zip(x_test, y_test)]) / len(x_test)

    @classmethod
    def test(cls):
        svm = cls(kernel="linear", max_iteration=200)
        x_train, x_test, y_train, y_test = \
            train_test_split(IRIS_DATASET.data[:100],
                             IRIS_DATASET.target[:100], test_size=0.25)
        svm.train(x_train, y_train)
        print(svm.alpha)
        print(svm.prediction_error_cache)
        from sklearn.metrics import accuracy_score
        print(accuracy_score(y_test, [svm.predict(x) for x in x_test]))
        print(svm.score(x_test, y_test))


if __name__ == "__main__":
    SupportVectorMachine.test()
