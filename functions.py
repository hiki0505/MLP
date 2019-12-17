import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Functions:
    # function that converts labels to one-hot array
    def to_one_hot(Y):
        n_col = np.amax(Y) + 1
        binarized = np.zeros((len(Y), n_col))
        for i in range(len(Y)):
            binarized[i, Y[i]] = 1.
        return binarized

    # function that converts one-hot array to labels
    # def from_one_hot(Y):
    #     arr = np.zeros((len(Y), 1))
    #
    #     for i in range(len(Y)):
    #         l = output_layer[i]
    #         for j in range(len(l)):
    #             if (l[j] == 1):
    #                 arr[i] = j + 1
    #     return arr

    # activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # derivative of activation function
    def sigmoid_deriv(x):
        return np.exp(-x)*(1 / (np.exp(-x) + 1)**2)

    # normalization of data
    def normalize(X, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1
        return X / np.expand_dims(l2, axis)
