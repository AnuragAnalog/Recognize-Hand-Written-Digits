#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Custom Modules
from metrics import MSE
from encoding import OneHotEncoder
from activation_function import sigmoid, softmax

class Classifier():
    def __init__(self, features, labels, hidden_layer=[], learning_rate=0.01):
        self.model_ran = False
        self.epochs = 10
        self.lr = learning_rate
        self.input = features
        self.hidden_layers = hidden_layer
        self.output = labels
        self.layers = len(hidden_layer) + 2
        self.nodes = [self.input]
        for node in self.hidden_layers:
            self.nodes = self.nodes + [node]
        self.nodes = self.nodes + [self.output]
        self.__initialize_weights()
        self.zero_grad()
        self.a = dict()

    def __str__(self):
        about_hyperp = "Epochs: "+str(self.epochs)+", Learning Rate: "+str(self.lr)
        about_layers = ", Layers: "+str(self.layers)
        return "Classifier("+about_hyperp+about_layers+")\n"

    def __initialization(self, row, col, variant='he', bias=False):
        # Initializing the weight matrix with random values.

        if variant == 'he':
            w = np.random.randn(row, col) * np.sqrt(2 / row)
        elif variant == 'xavier':
            w = np.random.randn(row, col) * np.sqrt(1 / row)
        elif variant == 'random':
            w = np.random.randn(row, col)
        elif variant == 'zero':
            w = np.zeros((row, col))

        if bias:
            b = np.random.randn((col, 1))
        else:
            b = np.zeros((col, 1))

        return w, b

    def __initialize_weights(self):
        self.W = dict()
        self.b = dict()

        for n in range(self.layers-1):
            w, b = self.__initialization(self.nodes[n], self.nodes[n+1])
            self.W["W"+str(n+1)], self.b["b"+str(n+1)] = w, b

        return

    def __reshape_input(self, vector):
        if isinstance(vector, list):
            vector = np.array(vector)

        vector = vector.reshape((self.input, -1))

        return vector

    def __reshape_output(self, vector):
        if isinstance(vector, list):
            vector = np.array(vector)

        vector = vector.reshape((self.output, -1))

        return vector

    def forward(self, x):
        self.a = [x]

        for n in range(self.layers-1):
            x = sigmoid(np.dot(self.W['W'+str(n+1)].T, x) + self.b['b'+str(n+1)])
            self.a.append(x)

        return x

    def backward(self, x, actual, output):
        self.da = dict()
        self.dz = dict()

        for n in range(self.layers-1, 0, -1):
            if n == self.layers-1:
                self.da['da'+str(n)] = (actual - self.a[n])
            else:
                self.da['da'+str(n)] = np.dot(self.dW['dW'+str(n+1)], self.dz['dz'+str(n+1)])

            self.dz['dz'+str(n)] = self.da['da'+str(n)] * (self.a[n] * (1 - self.a[n]))
            self.dW['dW'+str(n)] += np.dot(self.dz['dz'+str(n)], self.a[n-1].T).T/self.data_size
            self.db['db'+str(n)] += self.dz['dz'+str(n)]/self.data_size

        return

    def update_weights(self):
        for n in range(self.layers-1):
            self.W['W'+str(n+1)] += self.lr * self.dW['dW'+str(n+1)]
            self.b['b'+str(n+1)] += self.lr * self.db['db'+str(n+1)]

        return

    def model(self, x, y, epochs=100, debug=False, debug_verbose=False):
        self.model_ran = True
        self.training_loss = list()
        self.epochs = epochs
        self.data_size = len(x)

        for e in range(self.epochs):
            predict = list()
            self.zero_grad()
            for i in range(self.data_size):
                # Forward pass
                x_tmp = self.__reshape_input(x[i])
                output = self.forward(x_tmp)
                predict.append(output)
                y_tmp = self.__reshape_output(y[i])
                self.backward(x_tmp, y_tmp, output)
            self.update_weights()

            if debug:
                self.debug(more_verbose=debug_verbose)
            predict = np.array(predict)
            self.training_loss.append(np.squeeze(MSE(y, predict)))
            print(f"Epochs: {e+1}/{self.epochs}.. Training Loss: {self.training_loss[e]}..")

    def zero_grad(self):
        self.dW = dict()
        self.db = dict()

        for n in range(self.layers-1):
            self.dW['dW'+str(n+1)] = np.zeros((self.nodes[n], self.nodes[n+1]))
            self.db['db'+str(n+1)] = np.zeros((self.nodes[n+1], 1))

        return

    def debug(self, more_verbose):
        pass

    def get_weights(self):
        return self.W

    def get_biases(self):
        return self.b

    def summary(self):
        if self.model_ran is False:
            print("Run the model to view summary")
            return

    def epoch_vs_error(self, savefig=False):
        if self.model_ran is False:
            print("Run the model to view the graph")
            return

        plt.figure()

        plt.grid(True)
        plt.xlabel("Epochs")
        plt.ylabel("Error")

        plt.plot(np.arange(self.epochs)+1, self.training_loss)
        if savefig:
            plt.savefig('epoch-vs-error.png')
        plt.show()

        return

    def predict(self, y_new):
        if len(y_new) != self.input:
            print("Dimension mismatch")
            sys.exit()

        vector = self.__reshape_input(y_new)
        predicted = self.forward(vector)

        return predicted

if __name__ == '__main__':
    fname = 'mnist_train.csv'
    data = pd.read_csv(fname, nrows=400)

    ### Preproscessing
    scale = MinMaxScaler()
    encode = OneHotEncoder()

    encode.fit(data.label)
    labels = encode.transform(data.label)

    data.drop(['label'], axis=1, inplace=True)
    scale.fit(data)
    data = scale.transform(data)

    # x = data.values
    x = data
    y = labels

    ### Running the model
    network = Classifier(784, 10, [32], learning_rate=0.1)
    network.model(x, y, 100)
    network.epoch_vs_error(savefig=True)
    print(network.predict(x[5]), y[5])
    # print(network.activations)