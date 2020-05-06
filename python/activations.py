#!/usr/bin/python3

import numpy as np

class Activation():
    def __init__(self, function='sigmoid'):
        self.function = self.sigmoid
        if function == 'tanh':
            self.function = self.tanh
        elif function == 'arctan':
            self.function = self.arctan
        elif function == 'relu':
            self.function = self.relu
        elif function == 'leakyrelu':
            self.function = self.leakyrelu
        elif function == 'swish':
            self.function = self.swish
        elif function == 'softmax':
            self.function = self.softmax

    def sigmoid(self, z, derivative=False):
        a = 1 / (1 + np.exp(-z))
    
        if derivative:
            return a * (1 - a)
        return a
    
    def tanh(self, z, derivative=False):
        a = np.tanh(z)
    
        if derivative:
            return (1 - a**2)
        return a
    
    def arctan(self, z, derivative=False):
        a = np.arctan(z)
    
        if derivative:
            return 1/(z**2 + 1)
        return a
    
    def relu(self, z, derivative=False):
        a = np.maximum(np.zeros_like(z), z)
    
        if derivative:
            return (a > 0).astype(int)
        return a
    
    def leakyrelu(self, z, derivative=False):
        a = np.maximum(0.01*z, z)
    
        if derivative:
            return np.where(z > 0, 1, 0.01)        
        return a
    
    def swish(self, z, derivative=False):
        a = z / (1 + np.exp(-z))
    
        if derivative:
            return a + (1 / (1 + np.exp(-z))) * (1 - a)
        return a
    
    def softmax(self, z, derivative=False):
        a = np.exp(z)/(np.sum(np.exp(z)))

        if derivative:
            pass
        return a