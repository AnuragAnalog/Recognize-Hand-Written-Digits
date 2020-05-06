#!/usr/bin/python3

import numpy as np

def sigmoid(z, derivative=False):
    a = 1 / (1 + np.exp(-z))

    if derivative:
        return a * (1 - a)
    return a

def tanh(z, derivative=False):
    a = np.tanh(z)

    if derivative:
        return (1 - a**2)
    return a

def arctan(z, derivative=False):
    a = np.arctan(z)

    if derivative:
        return 1/(z**2 + 1)
    return a

def relu(z, derivative=False):
    a = np.maximum(np.zeros_like(z), z)

    if derivative:
        return (a > 0).astype(int)
    return a

def leakyrelu(z, derivative=False):
    a = np.maximum(0.01*z, z)

    if derivative:
        return np.where(z > 0, 1, 0.01)        
    return a

def softmax(x, derivative=False):
    x = np.array(x)
    dist = np.exp(x)/(np.sum(np.exp(x)))

    return dist