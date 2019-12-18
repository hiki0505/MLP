import numpy as np


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
        return np.exp(-x) * (1 / (np.exp(-x) + 1) ** 2)

    # normalization of data
    def normalize(X, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1
        return X / np.expand_dims(l2, axis)

    def confusion_matrix(y_hat, y_true):

        # Tp -- Fp -- Fn -- Tn
        matrix = [0, 0, 0, 0]
        # print(y_pred)
        for i in range(len(y_hat)):
            if np.argmax(y_hat[i]) == 0 and np.argmax(y_true[i]) == 0:
                matrix[0] += 1  # true positive
            elif np.argmax(y_hat[i]) == 0 and np.argmax(y_true[i]) == 1:
                matrix[1] += 1  # false positive
            elif np.argmax(y_hat[i]) == 1 and np.argmax(y_true[i]) == 0:
                matrix[3] += 1  # true negative
            else:
                matrix[2] += 1  # false negative
        return np.array(matrix)

    def split_train_test(data, train_size=0.7):
        indexes = np.random.permutation(len(data))

        train = data[indexes[:int(len(data) * train_size)]]
        test = data[indexes[int(len(data) * train_size):]]

        return (train, test)