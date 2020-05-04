#!/usr/bin/python3

import numpy as np
import pandas as pd

data = pd.read_csv('mnist_train.csv')
y = data[['label']].values[:400]
data.drop(['label'], axis=1, inplace=True)
x = data.values[:400]

def forward(w, b, x, n):
    for i in range(n):
        pass
    pass

def backward(train_x, train_y):
    pass

if __name__ == "__main__":
    pass