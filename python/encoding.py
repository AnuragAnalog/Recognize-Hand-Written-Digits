#!/usr/bin/python3

import numpy as np

class OneHotEncoder():
    def __init__(self):
        self.unique = dict()
        self.fit_called = False
        self.row = 2
        self.column = 2

    def __str__(self):
        if self.fit_called:
            return "Encoding is: "+str(self.unique)
        else:
            return "call the fit method to initialize some parameters"

    def __encode(self, index, n):
        return [0 if i is not index else 1 for i in range(n)]

    def fit(self, x):
        index = 0
        self.fit_called = True
        unique_values = set(x)

        for value in unique_values:
            self.unique[value] = index
            index = index + 1
        self.row = len(x)
        self.column = index

        return

    def transform(self, x):
        encoded = list()
        for col in x:
            for key in self.unique.keys():
                if col == key:
                    encoded.append(self.__encode(self.unique[key], self.column))
                    break

        return np.array(encoded).reshape(self.row, self.column)

class categorize():
    def __init__(self):
        self.unique = dict()
        self.fit_called = False
        self.row = 2

    def __str__(self):
        if self.fit_called:
            return "Encoding is: "+str(self.unique)
        else:
            return "call the fit method to initialize some parameters"
    def fit(self, x):
        index = 0
        self.fit_called = True
        unique_values = set(x)

        for value in unique_values:
            self.unique[value] = index
            index = index + 1
        self.row = len(x)

        return

    def transform(self, x):
        encoded = list()
        for col in x:
            for key in self.unique.keys():
                if col == key:
                    encoded.append(self.unique[key])
                    break

        return np.array(encoded)