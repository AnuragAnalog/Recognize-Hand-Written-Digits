#!/usr/bin/python3

from math import log

def MSE(actual, predict):
    error = 0
    if type(actual) == type([]) or len(actual.shape) == 1:
        length = len(actual)
        for i in range(length):
            error += (actual[i] - predict[i])**2
    else:
        length = actual.shape[0] * actual.shape[1]
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                error += (actual[i][j] - predict[i][j])**2
    error = error / length
    return error
