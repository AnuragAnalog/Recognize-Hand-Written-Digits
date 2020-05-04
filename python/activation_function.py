#!/usr/bin/python3

import numpy as np

def step(x: [int, float], T: [int, float]=0) -> int:
    if x < T:
        return 0
    else:
        return 1

def sigmoid(x: [int, float]) -> float:
    return 1 / (1 + np.exp(-x))

def tanh(x: [int, float]) -> float:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def piece_wise_linear(x: [int, float]) -> [int, float]:
    if x > 0.5:
        return 1
    elif x >= -0.5 and x <= 0.5:
        return x + 0.5
    else:
        return 0

def relu(x: [int, float]) -> [int, float]:
    if x < 0:
        return x - x
    else:
        return x

def softmax(x: list) -> list:
    x = np.array(x)
    dist = np.exp(x)/(np.sum(np.exp(x)))

    return dist