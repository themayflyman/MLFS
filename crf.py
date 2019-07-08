#!/usr/bin/python
# -*- encoding: utf8 -*-

import re
import numpy as np


START = 'start'
STOP = 'stop'


class FeatureExtractor:
    def __init__(self):
        pass


class LinearChainCRFTagger:
    def __init__(self, feature_func=None):
        self.total_features = None
        self.total_tags = None

        self.transition_feature_func = None
        self.transition_feature_func_weights = None
        self.state_feature_func = None
        self.state_feature_func_weights = None

        self.transition_matrices = None

        self.forward_vectors = None
        self.backward_vectors = None

    @staticmethod
    def _extract_features(word):
        feature_list = []

        if not word:
            return feature_list

        if word[0].isupper():
            feature_list.append('CAPITALIZATION')

        if re.search(re.compile('/d'), word) is not None:
            feature_list.append('HAS_NUM')

        import string
        punctuation = string.punctuation
        if list([w for w in word if w in punctuation]) is not None:
            feature_list.append('PUNCTUATION')

        if len(word) > 1:
            feature_list.append('SUF_' + word[-1:])
        if len(word) > 2:
            feature_list.append('SUF_' + word[-2:])
        if len(word) > 3:
            feature_list.append('SUF_' + word[-3:])

        feature_list.append('WORD_' + word)

        return feature_list

    def _generate_transition_matrices(self, sent, tag):
        """Generate a list of M*M transition matrices, where M is the num of
           tags.

        Parameters
        ----------

        Returns
        -------
        transition_matrices : list
            A list of M*M transition matrices with elements of the form

            M_i(y', y|x) = exp(∑ weight * feature_func(y', y, x, i))
        """
        transition_matrices = [None]

        for i in range(1, len(sent)):
            transition_matrix = np.zeros((len(self.total_tags),
                                          len(self.total_tags)))
            from itertools import product
            for j, k in product(range(len(self.total_tags)),
                                range(len(self.total_tags))):
                if i == 1:
                    transition_matrix[j, k] = np.exp(
                        sum([sffw * sff(sent[i], sent[i])
                             for sffw, sff in zip(
                             self.state_feature_func_weights,
                             self.state_feature_func)]))
                else:
                    transition_matrix[j, k] = np.exp(
                        sum([tffw * tff(tag[i-1],
                                        tag[i],
                                        sent[i])
                             for tffw, tff in zip(
                                self.transition_feature_func_weights[j, k],
                                self.transition_feature_func[j, k])])
                        + sum([sffw * sff(tag[i], sent[i])
                               for sffw , sff in zip(
                                self.state_feature_func_weights,
                                self.state_feature_func)])
                    )
            transition_matrices.append(transition_matrix)

        return transition_matrices

    def _compute_forward_vectors(self, sent_train, tag_train):
        """Compute forward vectors using Forward Algorithm.

        Parameters
        ----------
        sent_train : array-like
        tag_train : array-like

        Returns
        -------
        forward_vectors : array-like
        """
        forward_vectors = np.zeros((len(sent_train), len(self.total_tags)))

        for i in range(len(sent_train)):
            if i == 0:
                if tag_train[i] == START:
                    forward_vectors[0] = np.ones(len(self.total_tags))
                else:
                    forward_vectors[0] = 0
            else:
                forward_vectors[i] = np.dot(forward_vectors[i-1].T,
                                            self.transition_matrices[i])

        return forward_vectors

    def _compute_backward_vectors(self, sent, tag):
        """Compute backward vectors using Backward Algorithm.

        Parameters
        ----------
        sent : array-like
        tag : array-like

        Returns
        -------
        backward_vectors : array-like
        """
        backward_vectors = np.zeros((len(sent), len(self.total_tags)))

        for i in range(len(sent)-1, 0, -1):
            if i == len(sent)-1:
                if tag[i] == STOP:
                    backward_vectors[i] = np.ones(len(self.total_tags))
                else:
                    backward_vectors[i] = 0
            else:
                backward_vectors[i] = np.dot(backward_vectors[i+1].T,
                                             self.transition_matrices[i+1])

        return backward_vectors

    def compute_total_feature_count(self, sent, tag):
        """Compute the total feature count, notated as T(x, y) in the
           references.

        Parameters
        ----------
        sent
        tag

        Returns
        -------

        """
        from itertools import chain
        return sum([
            tff(prev_tag, tag[i], sent[i])
            for i in range(1, len(sent))
            for prev_tag in self.total_tags
            for tff in chain.from_iterable(self.transition_feature_func[:, tag[i]])
        ]) + sum([
            sff(tag[i], sent[i])
            for i in range(1, len(sent))
            for sff in self.state_feature_func
        ])

    def partition_func(self):
        return np.dot(self.forward_vectors[-1], np.ones(len(self.total_tags)).T)

    def word2features(self, sent, i):
        word = sent[i]
        return self._extract_features(word)

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def pre_process(self, sent_train, tag_train):
        sent_train = [self.sent2features(sent) for sent in sent_train]
        for s, t in zip(sent_train, tag_train):
            s.insert(0, START)
            s.append(STOP)
            t = [self.total_tags.index(tt) for tt in t]
            t.insert(0, START)
            t.append(STOP)
        return sent_train, tag_train

    def fit(self, sent_train, tag_train, algorithm='iis'):
        text_train, tag_train = self.pre_process(sent_train, tag_train)

        from itertools import chain
        self.total_features = set(
            chain.from_iterable(chain.from_iterable(text_train)))
        self.total_tags = set(chain.from_iterable(tag_train))

        self.transition_feature_func_weights = np.zeros(
            (len(self.total_tags), len(self.total_tags),
             len(self.total_features)))
        self.transition_feature_func = [[[
            lambda prev_tag, tag, word,
            feature=feature,
            expected_prev_tag=expected_prev_tag,
            expected_tag=expected_tag:
            1 if feature in word and (prev_tag, tag) == (expected_prev_tag,
                                                         expected_tag)
            else 0
            for feature in self.total_features]
            for expected_prev_tag in self.total_tags]
            for expected_tag in self.total_tags]

        self.state_feature_func_weights = np.zeros((
            len(self.total_tags), len(self.total_features)))
        self.state_feature_func = [[
            lambda tag, word, expected_tag=expected_tag, f=f:
            1 if f in word and tag == expected_tag
            else 0
            for f in self.total_features]
            for expected_tag in self.total_tags]

        # self.transition_matrices = self._generate_transition_matrices(
        #     text_train, tag_train)
        # self.forward_vectors = self._compute_forward_vectors(text_train,
        #                                                      tag_train)
        # self.backward_vectors = self._compute_backward_vectors(text_train,
        #                                                        tag_train)
        # print(self.transition_matrices)
        # print(self.forward_vectors)
        # print(self.backward_vectors)

        if algorithm == 'iis':
            fit_lccrf_with_iis(self, sent_train, tag_train)
        elif algorithm == 'gis':
            fit_lccrf_with_gis(self, sent_train, tag_train)


def fit_lccrf_with_iis(lccrftagger: LinearChainCRFTagger, text_train, tag_train):
    pass


def fit_lccrf_with_gis(lccrf_tagger: LinearChainCRFTagger,
                       sent_train, tag_train, max_iteration=100):
    pass


if __name__ == '__main__':
    pass
