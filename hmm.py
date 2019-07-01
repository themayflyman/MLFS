#!/usr/bin/python
# -*- encoding: utf8 -*-

import numpy as np


# TODO: A general logger for all Estimator.
class HiddenMarkovModel:
    """Implementations of Hidden Markov Model.

    Parameters
    ----------
    initial_state_prob : numpy.ndarray
        Initial state occupation distribution.

    transition_prob : numpy.ndarray
        Matrix of transition probabilities between states.

    emission_prob : numpy.ndarray
        Matrix of probabilities of an observation for a given state emission.

    observation_set : tuple
        a tuple of observations, observations are numbered by their order.

    state_set : tuple
        a tuple of states, states are numbered by their order.

    max_iteration: int, optional
        Maximum number of iterations to perform.

    criterion : float, optional
        Convergence threshold.

    Attributes
    ----------
    state_prob : numpy.ndarray
        State occupation distribution.

    transition_prob : numpy.ndarray
        Matrix of transition probabilities between states.

    emission_prob : numpy.ndarray
        Matrix of probabilities of an observation for a given state emission.

    state_set_num : int
        Number of states in the model.

    observation_set_num : int
        Number of the symbols of observations in the model.

    observation_len : int
        Length of the given observations sequence.

    forward_prob : numpy.ndarray
        Probabilities of seeing the given observations sequence, using the
        Forward Algorithm.

    backward_prob : numpy.ndarray
        Probabilities of seeing the given observations sequence, using the
        Backward Algorithm.

    Examples
    --------
    >>> initial_state_prob = np.array([.2, .4, .4])
    >>> transition_prob = np.array([[.5, .2, .3],
    ...                               [.3, .5, .2],
    ...                                [.2, .3, .5]])
    >>> emission_prob = np.array([[.5, .5],
    ...                              [.4, .6],
    ...                              [.7, .3]])
    >>> hmm = HiddenMarkovModel(initial_state_prob=initial_state_prob,
    ...                         transition_prob=transition_prob,
    ...                         emission_prob=emission_prob,
    ...                         observation_set=('红', '白'),
    ...                         state_set=(1, 2, 3))
    >>> hmm.fit(['红', '白', '红'])
    >>> print(hmm.decode(['红', '白', '红']))

    Notes
    -----

    References
    ----------
    """
    def __init__(self,
                 initial_state_prob,
                 transition_prob,
                 emission_prob,
                 observation_set,
                 state_set,
                 **kwargs):
        self.observation_set = observation_set
        self.observation_set_num = len(self.observation_set)

        self.state_set = state_set
        self.state_set_num = len(self.state_set)

        self.state_prob = initial_state_prob,
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob

        self.max_iteration = kwargs.get("max_iteration", 50)
        self.criterion = kwargs.get('criterion', 1e-3)

        self.observation_len = None

        self.forward_prob = None
        self.backward_prob = None

    def _compute_forward_prob(self, observations):
        """Compute the forward probabilities of seeing the given observations
           sequence under the model and its current parameters.

        Parameters
        ----------
        observations: array-like

        Returns
        -------
        forward_prob: array-like
            Forward probabilities of seeing the given observations sequence.
        """
        forward_prob = np.zeros((self.observation_len, self.state_set_num))

        forward_prob[0] = \
            self.state_prob \
            * self.emission_prob[:, observations[0]]

        for t in range(1, self.observation_len):
            for i in range(self.state_set_num):
                forward_prob[t][i] = sum(
                    [forward_prob[t-1][j] * self.transition_prob[j][i]
                     for j in range(self.state_set_num)]) \
                    * self.emission_prob[i][observations[t]]

        return forward_prob

    def _compute_backward_prob(self, observations):
        """Compute the backward probabilities of seeing the given observations
           sequence under the model and its current parameters.

        Parameters
        ----------
        observations: array-like

        Returns
        -------
        backward_prob: array-like
            Backward probabilities of seeing the given observations sequence.
        """
        backward_prob = np.zeros((self.observation_len, self.state_set_num))

        backward_prob[self.observation_len-1] = np.ones(self.state_set_num)

        for t in range(self.observation_len-2, -1, -1):
            for i in range(self.state_set_num):
                backward_prob[t][i] = sum(
                    self.transition_prob[i][j]
                    * self.emission_prob[j][observations[t+1]]
                    * backward_prob[t+1][j]
                    for j in range(self.state_set_num))

        return backward_prob

    def _compute_expected_state_transition_count(self, observations, t, i, j):
        """Compute the expected state transition count under the model and its
           current parameters.

        Parameters
        ----------
        observations : array-like
        t : int
            index of time t
        i : int
            index of state i
        j : int
            index of state j

        Returns
        -------
        expected_state_occupancy_count : float
            Probability of being in state i at time t and
            state j at time t+1.
        """
        return self.forward_prob[t][i] \
               * self.transition_prob[i][j] \
               * self.emission_prob[j][observations[t+1]] \
               * self.backward_prob[t+1][j] \
               / sum(sum(self.forward_prob[t][i]
                         * self.transition_prob[i][j]
                         * self.emission_prob[j][observations[t+1]]
                         * self.backward_prob[t+1][j]
                         for j in range(self.state_set_num))
                     for i in range(self.state_set_num))

    def _compute_expected_state_occupancy_count(self, t, i):
        """Compute the expected state occupancy count under the model and its
           current parameters.

        Parameters
        ----------
        t : int
            index of time t
        i : int
            index of state i

        Returns
        -------
        expected_state_transition_count : float
            Probability of being in state i at time t.
        """
        return self.forward_prob[t][i] * self.backward_prob[t][i] \
               / sum(self.forward_prob[t][j] * self.backward_prob[t][j]
                     for j in range(self.state_set_num))

    def _compute_log_likelihood(self, observations):
        pass

    def fit(self, observations):
        """Fit the model to the given observations sequence.

        Parameters
        ----------
        observations : array-like
        """
        self.observation_len = len(observations)
        observations = [self.observation_set.index(observation)
                        for observation in observations]

        for _ in range(self.max_iteration):
            self.forward_prob = self._compute_forward_prob(observations)
            self.backward_prob = self._compute_backward_prob(observations)

            updated_state_prob = np.zeros(self.state_set_num)
            updated_transition_prob = np.zeros((self.state_set_num,
                                                self.state_set_num))
            updated_emission_prob = np.zeros((self.state_set_num,
                                              self.observation_set_num))

            for i in range(self.state_set_num):
                for j in range(self.state_set_num):
                    updated_transition_prob[i][j] = \
                        sum(self._compute_expected_state_transition_count(
                            observations, t, i, j)
                            for t in range(self.observation_len-1)) \
                        / sum(self._compute_expected_state_occupancy_count(t, i)
                              for t in range(self.observation_set_num-1))

            for j in range(self.state_set_num):
                for k in range(self.observation_set_num):
                    updated_emission_prob[j][k] = \
                        sum(self._compute_expected_state_occupancy_count(t, j)
                            for t in range(self.observation_len)
                            if observations[t] == k) \
                        / sum(self._compute_expected_state_occupancy_count(t, j)
                              for t in range(self.observation_len))

            for i in range(self.state_set_num):
                updated_state_prob[i] = \
                    self._compute_expected_state_occupancy_count(0, i)

            if np.max(abs(updated_state_prob-self.state_prob)) \
               + np.max(abs(updated_transition_prob-self.transition_prob)) \
               + np.max(updated_emission_prob-self.emission_prob) \
               < self.criterion * 3.0:
                break

            self.state_prob = updated_state_prob
            self.transition_prob = updated_transition_prob
            self.emission_prob = updated_emission_prob

    def decode(self, observations, decoder='viterbi'):
        """Decode the model with the given observations sequence and specified
           decoder.

        Parameters
        ----------
        observations : array-like
        decoder : string, optional
           Decoder the model uses to decode the observations sequence.

        Returns
        -------
        states : array-like
            A most likely state sequence of the given observatons sequence.
        """
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
                  max_iteration=50)
        hmm.fit(['红', '白', '红'])
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
    """Decode a Hidden Markov Model using Viterbi Algorithm.

    Parameters
    ----------
    hmm: HiddenMarkovModel
    observations: array-like

    Returns
    -------
    states : array-like
        A most likely state sequence of the given observatons sequence.
    """
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
