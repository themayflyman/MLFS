#!/usr/bin/python
# -*- encoding: utf8 -*-

import numpy as np


class HiddenMarkovModel:
    """

    Parameters
    ----------
    initial_state_probability_matrix : numpy.ndarray
    transition_probability_matrix : numpy.ndarray
    observation_probability_matrix : numpy.ndarray
    observation_set : tuple
        a tuple of observations, observations are numbered by their order

    state_set : tuple
        a tuple of states, states are numbered by their order

    Notes
    -----

    References
    ----------
    """
    def __init__(self,
                 observation_set,
                 state_set,
                 **kwargs):
        self.observation_set = observation_set
        self.observation_set_num = len(self.observation_set)

        self.state_set = state_set
        self.state_set_num = len(self.state_set)

        self.max_iteration = kwargs.get("max_iteration", 100)
        self.state_prob = kwargs.get("initial_state_prob",
                                     np.ones(self.state_set_num)
                                     / self.state_set_num)
        self.transition_prob = kwargs.get("transition_prob",
                                          np.ones((self.state_set_num,
                                                   self.state_set_num)))
        self.emission_prob = kwargs.get("emission_prob",
                                        np.ones((self.state_set_num,
                                                 self.observation_set_num)))

        self.observations = None
        self.observation_len = None

        self.forward_prob = None
        self.backward_prob = None

    def _compute_forward_prob(self):
        forward_prob = np.zeros((self.observation_len, self.state_set_num))

        forward_prob[0] = \
            self.state_prob \
            * self.emission_prob[:, self.observations[0]]

        for t in range(1, self.observation_len):
            for i in range(self.state_set_num):
                forward_prob[t][i] = sum(
                    [forward_prob[t-1][j] * self.transition_prob[j][i]
                     for j in range(self.state_set_num)]) \
                    * self.emission_prob[i][self.observations[t]]

        return forward_prob

    def _compute_backward_prob(self):
        backward_prob = np.zeros((self.observation_len, self.state_set_num))

        backward_prob[self.observation_len-1] = np.ones(self.state_set_num)

        for t in range(self.observation_len-2, -1, -1):
            for i in range(self.state_set_num):
                backward_prob[t][i] = sum(
                    self.transition_prob[i][j]
                    * self.emission_prob[j][self.observations[t+1]]
                    * backward_prob[t+1][j]
                    for j in range(self.state_set_num))

        return backward_prob

    def _compute_o(self, t, i, j):
        return self.forward_prob[t][i] \
               * self.transition_prob[i][j] \
               * self.emission_prob[j][self.observations[t+1]] \
               * self.backward_prob[t+1][j] \
               / sum(sum(self.forward_prob[t][i]
                         * self.transition_prob[i][j]
                         * self.emission_prob[j][self.observations[t+1]]
                         * self.backward_prob[t+1][j]
                         for j in range(self.state_set_num))
                     for i in range(self.state_set_num))

    def _compute_t(self, t, i):
        return self.forward_prob[t][i] * self.backward_prob[t][i] \
               / sum(self.forward_prob[t][j] * self.backward_prob[t][j]
                     for j in range(self.state_set_num))

    def learn(self, observations):
        self.observation_len = len(observations)
        self.observations = [self.observation_set.index(observation)
                             for observation in observations]

        # TODO: Add convergence term
        for _ in range(self.max_iteration):
            self.forward_prob = self._compute_forward_prob()
            self.backward_prob = self._compute_backward_prob()

            for i in range(self.state_set_num):
                for j in range(self.state_set_num):
                    self.transition_prob[i][j] = \
                        sum(self._compute_o(t, i, j)
                            for t in range(self.observation_len-1)) \
                        / sum(self._compute_t(t, i)
                              for t in range(self.observation_set_num-1))

            for j in range(self.state_set_num):
                for k in range(self.observation_set_num):
                    self.emission_prob[j][k] = \
                        sum(self._compute_t(t, j)
                            for t in range(self.observation_len)
                            if self.observations[t] == k) \
                        / sum(self._compute_t(t, j)
                              for t in range(self.observation_len))

    def decode(self, observations, decoder='viterbi'):
        observations = [self.observation_set.index(observation)
                        for observation in observations]

        if decoder == "viterbi":
            return decode_by_viterbi(self, observations)
        else:
            raise ValueError("Unsupported decoder: {}".format(decoder))

    @classmethod
    def test(cls):
        initial_state_prob = np.array([.25, .25, .25, .25])
        transition_prob = np.array([[0, 1, 0, 0],
                                    [.4, 0, .6, 0],
                                    [0, .4, 0, .6],
                                    [0, 0, .5, .5]])
        emission_prob = np.array([[.5, .5],
                                  [.3, .7],
                                  [.6, .4],
                                  [.8, .2]])
        initial_state_prob = np.array([.2, .4, .4])
        transition_prob = np.array([[.5, .2, .3],
                                    [.3, .5, .2],
                                    [.2, .3, .5]])
        emission_prob = np.array([[.5, .5],
                                  [.4, .6],
                                  [.7, .3]])
        hmm = cls(initial_state_prob=initial_state_prob,
                  transition_prob=transition_prob,
                  emission_prob=emission_prob,
                  observation_set=('红', '白'), state_set=(1, 2, 3),
                  max_iteration=1)
        hmm.learn(['红', '白', '红'])
        print(hmm.forward_prob)
        print(hmm.backward_prob)
        print(hmm.transition_prob)
        print(hmm.emission_prob)
        print(hmm.decode(['红', '白', '红']))


def viterbi_backtrack_best_path(observation_len,
                                best_path_prob,
                                best_path_pointer):
    last_state = np.argmax(best_path_prob[observation_len-1])
    yield last_state
    for t in range(observation_len-2, -1, -1):
        yield int(best_path_pointer[t, last_state])
        last_state = int(best_path_pointer[t, last_state])


def decode_by_viterbi(hmm, observations):
    observation_len = len(observations)
    path_prob = np.zeros((len(observations), hmm.state_set_num))
    path_pointer = np.zeros((len(observations), hmm.state_set_num))

    path_prob[:, 1] = \
        hmm.state_prob \
        * hmm.emission_prob[:, observations[1]]
    path_pointer[:, 1] = 0
    for t in range(1, len(observations)):
        for i in range(hmm.state_set_num):
            path_prob[t][i] = max([path_prob[t-1][j]
                                   * hmm.transition_prob[j][i]
                                   for j in range(hmm.state_set_num)]) \
                              * hmm.emission_prob[i][observations[t]]
            path_pointer[t][i] = \
                np.argmax([path_prob[t-1][j]
                           * hmm.transition_prob[j][i]
                           for j in range(hmm.state_set_num)])

    state_pointers = reversed(list(viterbi_backtrack_best_path(observation_len,
                                                               path_prob,
                                                               path_pointer)))
    return list(hmm.state_set[pointer] for pointer in state_pointers)


if __name__ == "__main__":
    HiddenMarkovModel.test()
