#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal


class GaussianMixtureModel:
    def __init__(self, num_of_sources, **kwargs):
        self.max_iteration = kwargs.get("max_iteration", 50)

        self.num_of_sources = num_of_sources
        self.weights = kwargs.get("weights")

        self.mean = None
        self.covariance = None
        self.fraction_per_class = None

        self.r = dict((c, []) for c in range(self.num_of_sources))
        self.log_likelihood = []

        self.observable_variables = None
        self.latent_variables = None

        self._n = None

    def expectation_step(self, observations):
        self.r = dict((c, []) for c in range(self.num_of_sources))
        for observation in observations:
            prob_of_observation_all_over_classes = sum(
                self.fraction_per_class[c] * multivariate_normal(
                    self.mean[c], self.covariance[c]).pdf(observation)
                for c in range(self.num_of_sources))
            for c in range(self.num_of_sources):
                prob_that_observation_belongs_to_class_c = \
                    self.fraction_per_class[c] * \
                    multivariate_normal(self.mean[c],
                                        self.covariance[c]).pdf(observation)
                r = prob_that_observation_belongs_to_class_c / \
                    prob_of_observation_all_over_classes
                self.r[c].append(r)

    def maximization_step(self, observations):
        for c in range(self.num_of_sources):
            self.weights[c] = sum(r for r in self.r[c])

            self.fraction_per_class[c] = self.weights[c] / sum(self.r[c])

            self.mean[c] = \
                sum(r * o for r, o in zip(self.r[c], observations)) / \
                self.weights[c]

            # self.covariance[c] = \
            #     sum(np.dot(r * (o - m).T, (o - m))
            #         for r, o, m in zip(self.r[c],
            #                            observations,
            #                            self.mean[c])) / self.weights[c]
            self.covariance[c] = \
                np.dot(np.array(self.r[c]).T * (observations - self.mean[c]).T,
                       (observations - self.mean[c])) / self.weights[c]

    def fit(self, observations):
        self._n = observations.shape[1]
        self.mean = np.zeros((self.num_of_sources, self._n))
        self.covariance = dict((c, np.zeros((self._n, self._n)))
                               for c in range(self.num_of_sources))
        for c in range(self.num_of_sources):
            np.fill_diagonal(self.covariance[c], 5)
        self.fraction_per_class = \
            np.ones(self.num_of_sources) / self.num_of_sources
        self.weights = dict((c, list()) for c in range(self.num_of_sources))
        for _ in range(self.max_iteration):
            self.expectation_step(observations)
            self.maximization_step(observations)
            print(self.mean, self.covariance, self.fraction_per_class)
            log_likelihood = sum(np.log(sum(self.fraction_per_class[c] * multivariate_normal(self.mean[c], self.covariance[c]).pdf(observation)
                                            for c in range(self.num_of_sources)))
                                        for observation in observations)
            self.log_likelihood.append(log_likelihood)
            # check if converges

    @classmethod
    def test(cls):
        x_train, y_train = make_blobs(cluster_std=1.5,
                                      random_state=20,
                                      n_samples=500,
                                      centers=3)
        gmm = cls(num_of_sources=3)
        gmm.fit(x_train)
        print(gmm.log_likelihood)


if __name__ == "__main__":
    GaussianMixtureModel.test()
