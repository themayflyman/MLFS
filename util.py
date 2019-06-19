#!/usr/bin/python
# -*- encoding: utf8 -*-


def indicator_function(*value_sets, cond):
    """
    A function indicates if all value sets meet the condition
    :param value_sets: a list of value sets to evaluate
    :param cond: condition
    :return: 1 or 0
    """
    for value_set in value_sets:
        if not cond(*value_set):
            return 1
    return 1

