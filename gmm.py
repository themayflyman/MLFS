#!/usr/bin/python
# -*- encoding: utf8 -*-

from copy import deepcopy
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


# All computation is done via numpy matrix to be faster
class GaussianMixtureModel:
    def __init__(self, **kwargs):
        self.max_iteration = kwargs.get("max_iteration", 50)
        self.cluster_num = 0

        self.initial_mean = None
        self.initial_covariance = None
        self.initial_fraction_per_class = None

        self.mean = None
        self.covariance = None
        self.fraction_per_class = None

        self.regularization_of_covariance = None

        self.log_likelihood = []

    def fit(self, observations, cluster_num):
        self.cluster_num = cluster_num

        self.initial_mean = \
            np.random.randint(min(observations[:, 0]),
                              max(observations[:, 0]),
                              size=(cluster_num, observations.shape[1]),)
        self.initial_mean = np.float64(self.initial_mean)
        self.mean = deepcopy(self.initial_mean)

        self.initial_covariance = np.zeros((cluster_num,
                                            observations.shape[1],
                                            observations.shape[1]))

        for d in range(self.initial_covariance.shape[0]):
            np.fill_diagonal(self.initial_covariance[d], 5)

        self.covariance = deepcopy(self.initial_covariance)

        self.initial_fraction_per_class = np.ones(cluster_num) / cluster_num
        self.fraction_per_class = deepcopy(self.initial_fraction_per_class)

        self.regularization_of_covariance = 1e-6 * np.identity(observations
                                                               .shape[1])

        for _ in range(self.max_iteration):
            """Expectation Step"""
            # initialize the container of the probability that each observation
            # belongs to each cluster
            r = np.zeros((observations.shape[0], self.covariance.shape[0]))
            prob_of_observations_all_over_clusters = np.sum(
                [fraction * multivariate_normal(mean, cov).pdf(observations)
                 for fraction, mean, cov in zip(
                    self.fraction_per_class,
                    self.mean,
                    self.covariance+self.regularization_of_covariance)],
                axis=0
            )
            for cluster_id in range(self.cluster_num):
                prob_that_observations_belong_to_the_cluster = \
                    self.fraction_per_class[cluster_id] \
                    * multivariate_normal(
                        self.mean[cluster_id],
                        self.covariance[cluster_id]
                        + self.regularization_of_covariance)\
                    .pdf(observations)
                r[:, cluster_id] = \
                    prob_that_observations_belong_to_the_cluster \
                    / prob_of_observations_all_over_clusters

            """Maximization Step"""
            self.mean = []
            for cluster_id in range(self.cluster_num):
                weight = np.sum(r[:, cluster_id], axis=0)
                self.mean.append(
                    np.sum(observations * r[:, cluster_id]
                           .reshape(observations.shape[0], 1), axis=0) / weight)
                self.covariance[cluster_id] = \
                    (np.dot(
                        np.array(r[:, cluster_id]
                                 .reshape(observations.shape[0], 1) *
                                 (observations - self.mean[cluster_id])).T,
                        (observations - self.mean[cluster_id])) / weight) \
                    + self.regularization_of_covariance
                self.fraction_per_class[cluster_id] = weight / np.sum(r)

            log_likelihood = np.log(
                np.sum(
                    [fraction * multivariate_normal(mean, cov).pdf(observations)
                     for fraction, mean, cov in zip(self.fraction_per_class,
                                                    self.mean,
                                                    self.covariance)]
                )
            )
            self.log_likelihood.append(log_likelihood)

    def plot(self, observations):
        o, c = np.meshgrid(np.sort(observations[:, 0]),
                           np.sort(observations[:, 1]))

        plt.figure(figsize=(20, 10))

        plt.subplot(121)
        plt.title("Initial State")
        plt.scatter(observations[:, 0], observations[:, 1])
        for mean, cov in zip(self.initial_mean, self.initial_covariance):
            plt.contour(np.sort(observations[:, 0]),
                        np.sort(observations[:, 1]),
                        multivariate_normal(
                            mean,
                            cov+self.regularization_of_covariance)
                        .pdf(np.array([o.flatten(), c.flatten()]).T)
                        .reshape(observations.shape[0], observations.shape[0]),
                        alpha=0.3)
            plt.scatter(*mean, c='grey', zorder=10, s=100)

        plt.subplot(122)
        plt.title("Final State")
        plt.scatter(observations[:, 0], observations[:, 1])
        for mean, cov in zip(self.mean, self.covariance):
            plt.contour(np.sort(observations[:, 0]),
                        np.sort(observations[:, 1]),
                        multivariate_normal(mean, cov)
                        .pdf(np.array([o.flatten(), c.flatten()]).T)
                        .reshape(observations.shape[0], observations.shape[0]),
                        alpha=0.3)
            plt.scatter(*mean, c='grey', zorder=10, s=100)

        plt.figure(figsize=(10, 10))
        plt.subplot(111)
        plt.title('Log-Likelihood')
        plt.plot(range(0, self.max_iteration, 1), self.log_likelihood)

        plt.show()

    def predict(self, observation):
        prediction = dict().fromkeys(range(self.cluster_num))
        for cluster, mean, covariance in zip(range(self.cluster_num),
                                             self.mean,
                                             self.covariance):
            prediction[cluster] = \
                multivariate_normal(mean, covariance).pdf(observation) \
                / np.sum([
                    fraction * multivariate_normal(mean, cov).pdf(observation)
                    for fraction, mean, cov in zip(
                        self.fraction_per_class,
                        self.mean,
                        self.covariance+self.regularization_of_covariance)],
                    axis=0
                )
        return prediction

    @classmethod
    def test(cls):
        observations, clusters = make_blobs(cluster_std=1.5,
                                            random_state=20,
                                            n_samples=500,
                                            centers=3)
        observations = np.dot(observations,
                              np.random.RandomState(0).randn(2, 2))
        gmm = cls(max_iteration=60)
        gmm.fit(observations, 3)
        gmm.plot(observations)
        print(gmm.predict([0.5, 0.5]))


if __name__ == "__main__":
    GaussianMixtureModel.test()
