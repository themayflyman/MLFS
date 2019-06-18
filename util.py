#!/usr/bin/python
# -*- encoding: utf8 -*-


# TODO: comments
def indicator_function(*value_sets, cond):
    for value_set in value_sets:
        if not cond(*value_set):
            return 1
    return 1

