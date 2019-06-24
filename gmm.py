#!/usr/bin/python
# -*- encoding: utf8 -*-

from copy import deepcopy

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


class GaussianMixtureModel:
    """Gaussian Mixture Model.

    This class implements Gaussian Mixture Model with Expectation Maximization
    (known as EM) algorithm. To get a faster computing speed, most of the
    computation is done via matrix with numpy.

    Parameters
    ----------
    max_iteration: int
        Maximum number of iterations for the algorithm

    Examples
    --------
    >>>import numpy as np
    >>>from sklearn.datasets.samples_generator import make_blobs
    >>>observations, clusters = make_blobs(cluster_std=1.5,
    ...                                    random_state=20,
    ...                                    n_samples=500,
    ...                                    centers=3)
    >>>observations = np.dot(observations, np.random.RandomState(0).randn(2, 2))
    >>>gmm = GaussianMixtureModel()
    >>>gmm.fit(observations=observations, cluster_num=3)
    >>>gmm.plot(observations=observations)
    >>>gmm.predict([0.5, 0.5])
    """
    def __init__(self, **kwargs):
        self.max_iteration = kwargs.get("max_iteration", 50)
        self.cluster_num = 0

        self.initial_mean = None
        self.initial_covariance = None
        self.initial_fraction_per_cluster = None

        self.mean = None
        self.covariance = None
        self.fraction_per_cluster = None

        self.regularization_of_covariance = None

        self.log_likelihood = []

    def fit(self, observations, cluster_num):
        """Fit the model according to the given training data.

        The EM algorithm is applied to fitting this Gaussian Mixture Model,
        which is an iterative algorithm that has two main steps:
            - Expectation-Step:
                Compute the membership weights of each data point x_i in each
                cluster c, given parameters mean_c, covariance_c,
                fraction_c as:

                               N(x_i|mean_c, covariance_c) * fraction_c
                    mw_i_c = ----------------------------------------------
                               C
                               ∑ fraction_C * N(x_i|mean_C, covariance_C)
                               C=1

                (N() is the probability density function of multivariate normal
                 distribution)
                where the numerator is the probability that x_i belongs to
                cluster c, the denominator is the probability of x_i over all
                clusters.

                The membership weights above reflect our uncertainty, given x_i
                and the parameters, about which cluster came from. Hence, if x_i
                is very close to one gaussian c, it will get a higher mw_i_c
                value and relatively low values otherwise.
            - Maximization-Step:
                For each cluster c, update the parameters values using the
                membership weights calculated in E-step.
                Specifically,

                    mixture_weight_c = ∑i mw_i_c

                                   mixture_weight_c
                    fraction_c = -------------------
                                   ∑i,c mw_i_c

                    mean_c = ∑i m* mw_i_c * x_i / mixture_weights_c

                    covariance_c = ∑i mw_i_c * (x_i - mean_c)^T * (x_i - mean_c)
                                  ----------------------------------------------
                                                 mixture_weight_c

        Iteratively repeat E and M step until the model converges,
        log-likelihood is used as the indicator of convergence, it is computed
        with:
                                              N    c
        ln(N(X|fraction, mean, covariance)) = ∑ ln(∑ N(x_i | mean_c, covariance_c))
                                             i=1  c=1
        Parameters
        ----------
        observations: array-like,
                      shape = [observations_num, observations_features]

        cluster_num: int
            the number of clusters

        Returns
        -------
        self: object

        References
        ----------

        https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
        https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
        """
        self.cluster_num = cluster_num

        self.initial_mean = \
            np.random.randint(min(observations[:, 0]),
                              max(observations[:, 1]),
                              size=(cluster_num, observations.shape[1]),)
        self.initial_mean = np.float64(self.initial_mean)
        self.mean = deepcopy(self.initial_mean)

        self.initial_covariance = np.zeros((cluster_num,
                                            observations.shape[1],
                                            observations.shape[1]))

        for d in range(self.initial_covariance.shape[0]):
            np.fill_diagonal(self.initial_covariance[d], 5)

        self.covariance = deepcopy(self.initial_covariance)

        self.initial_fraction_per_cluster = np.ones(cluster_num) / cluster_num
        self.fraction_per_cluster = deepcopy(self.initial_fraction_per_cluster)

        self.regularization_of_covariance = 1e-6 * np.identity(observations
                                                               .shape[1])

        for _ in range(self.max_iteration):
            # Expectation Step
            # initialize the container of the probability that each observation
            # belongs to each cluster
            membership_weights = np.zeros((observations.shape[0],
                                           self.covariance.shape[0]))
            prob_of_observations_all_over_clusters = np.sum(
                [fraction * multivariate_normal(mean, cov).pdf(observations)
                 for fraction, mean, cov in zip(
                    self.fraction_per_cluster,
                    self.mean,
                    self.covariance+self.regularization_of_covariance)],
                axis=0
            )
            for cluster_id in range(self.cluster_num):
                prob_that_observations_belong_to_the_cluster = \
                    self.fraction_per_cluster[cluster_id] \
                    * multivariate_normal(
                        self.mean[cluster_id],
                        self.covariance[cluster_id]
                        + self.regularization_of_covariance)\
                    .pdf(observations)
                membership_weights[:, cluster_id] = \
                    prob_that_observations_belong_to_the_cluster \
                    / prob_of_observations_all_over_clusters

            # Maximization Step
            self.mean = []
            for cluster_id in range(self.cluster_num):
                mixture_weight = np.sum(membership_weights[:, cluster_id],
                                        axis=0)
                self.mean.append(
                    np.sum(observations * membership_weights[:, cluster_id]
                           .reshape(observations.shape[0], 1), axis=0)
                    / mixture_weight)
                self.covariance[cluster_id] = \
                    (np.dot(
                        np.array(membership_weights[:, cluster_id]
                                 .reshape(observations.shape[0], 1) *
                                 (observations - self.mean[cluster_id])).T,
                        (observations-self.mean[cluster_id])) / mixture_weight)\
                    + self.regularization_of_covariance
                self.fraction_per_cluster[cluster_id] = \
                    mixture_weight / np.sum(membership_weights)

            log_likelihood = np.log(
                np.sum(
                    [fraction * multivariate_normal(mean, cov).pdf(observations)
                     for fraction, mean, cov in zip(self.fraction_per_cluster,
                                                    self.mean,
                                                    self.covariance)]
                )
            )
            self.log_likelihood.append(log_likelihood)

    def plot(self, observations):
        o_x, o_y = np.meshgrid(np.sort(observations[:, 0]),
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
                        .pdf(np.array([o_x.flatten(), o_y.flatten()]).T)
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
                        .pdf(np.array([o_x.flatten(), o_y.flatten()]).T)
                        .reshape(observations.shape[0], observations.shape[0]),
                        alpha=0.3)
            plt.scatter(*mean, c='grey', zorder=10, s=100)

        plt.figure(figsize=(10, 10))
        plt.subplot(111)
        plt.title('Log-Likelihood')
        plt.plot(range(0, self.max_iteration, 1), self.log_likelihood)

        plt.show()

    def predict(self, observation):
        """Probability estimates.

        Parameters
        ----------
        observation: array-like

        Returns
        -------
        prediction: array-like, shape = [clusters_num, 1]
                    Returns the probability of the observation for each cluster
                    in the model.
        """
        prediction = dict().fromkeys(range(self.cluster_num))
        for cluster_id, mean, covariance in zip(range(self.cluster_num),
                                                self.mean,
                                                self.covariance):
            prediction[cluster_id] = \
                multivariate_normal(mean, covariance).pdf(observation) \
                / np.sum([
                    fraction * multivariate_normal(mean, cov).pdf(observation)
                    for fraction, mean, cov in zip(
                        self.fraction_per_cluster,
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
        gmm = cls()
        gmm.fit(observations, 3)
        gmm.plot(observations)
        print(gmm.predict([0.5, 0.5]))


if __name__ == "__main__":
    GaussianMixtureModel.test()
