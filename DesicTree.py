import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split


class decision_tree():
    class __rule():
        def __init__(self, col, val):
            self.__col = col
            self.__val = val

        def is_satisfied(self, data):
            if self.__col == None and self.__val == None:
                return True

            return data[0, self.__col] == self.__val

        def __str__(self):
            return f"col: {self.__col}, val: {self.__val}"

    def __init__(self, data, n_groups=3, max_depth=5, indexes="all", cols_exclude=[], rule=None, groups=[]):
        self.__rule_ = rule
        self.__max_depth = max_depth
        self.__data = data
        self.__branches = []
        self.__cols_exclude = cols_exclude
        self.__n_groups = n_groups
        self.__groups = groups

        if isinstance(indexes, str) and indexes == "all":
            self.__indexes = np.ones(len(data[0]), dtype=bool)
        else:
            self.__indexes = indexes.copy()

        self.float_to_one_hot()

        if max_depth > 0:
            self.__branches = self.__branch()

    def get_rule(self):
        return self.__rule_

    def get_data(self):
        return self.__data

    def get_groups(self):
        return self.__groups

    def float_to_one_hot(self, data=[]):

        if len(data) == 0:
            if len(self.__groups) != 0:
                return self.__data[0], self.__groups
            data = self.__data[0].copy()
            groups = np.zeros((self.__data[0].shape[1], self.__n_groups))

            for col in range(data.shape[1]):
                data = data[data[:, col].argsort()]
                for i in range(self.__n_groups):
                    cur_val = data[(i + 1) * data.shape[0] // self.__n_groups - 1, col]
                    groups[col, i] = cur_val

            new_d = np.empty(data.shape)

            for col in range(groups.shape[0]):
                for i in range(groups.shape[1]):
                    if i == 0:
                        new_d[:, col][self.__data[0][:, col] <= groups[col, i]] = i
                    else:
                        new_d[:, col][(self.__data[0][:, col] <= groups[col, i]) & (
                                    self.__data[0][:, col] > groups[col, i - 1])] = i
                new_d[:, col][self.__data[0][:, col] > groups[col, -1]] = self.__n_groups - 1

            self.__groups = groups.copy()
            self.__data = (new_d.copy(), self.__data[1])

        else:
            groups = self.__groups
            new_d = np.empty(data.shape)
            for col in range(groups.shape[0]):
                for i in range(groups.shape[1]):
                    if i == 0:
                        new_d[:, col][data[:, col] <= groups[col, i]] = i
                    else:
                        new_d[:, col][(data[:, col] <= groups[col, i]) & (data[:, col] > groups[col, i - 1])] = i
                new_d[:, col][data[:, col] > groups[col, -1]] = self.__n_groups - 1

        return new_d, groups

    def __branch(self):
        if len(np.unique(self.__data[1][self.__indexes])) <= 1:
            return []

        branches = []
        disc_max = -np.inf
        disc_max_col = None

        for col in range(self.__data[0].shape[1]):
            if col in self.__cols_exclude:
                continue
            disc = self.disc(col)
            if disc > disc_max:
                disc_max = disc
                disc_max_col = col

        if disc_max_col == None:
            return branches

        uniques = np.unique(self.__data[0][self.__indexes, disc_max_col])
        cols_exclude = [col for col in self.__cols_exclude]
        cols_exclude.append(disc_max_col)
        for unique in uniques:
            indexes = (self.__data[0][:, disc_max_col] == unique)
            indexes = np.logical_and(self.__indexes, indexes)
            rule = self.__rule(disc_max_col, unique)
            branches.append(
                decision_tree(self.__data, self.__n_groups, self.__max_depth - 1, indexes, cols_exclude, rule,
                              self.__groups))

        return branches

    def entropy(self, col=None):
        res = 0
        if col == None:
            uniques, counts = np.unique(self.__data[1][self.__indexes], return_counts=True)
            for val, count in zip(uniques, counts):
                if len(self.__data[1][self.__indexes] == val) != 0:
                    res -= (count / len(self.__data[1][self.__indexes]) * np.log2(
                        count / len(self.__data[1][self.__indexes])))
        else:
            uniques_y = np.unique(self.__data[1][self.__indexes], return_counts=True)
            uniques_x, counts_x = np.unique(self.__data[0][self.__indexes, col], return_counts=True)
            for unique_x, count_x in zip(uniques_x, counts_x):
                y_indexes = np.logical_and(self.__indexes, (self.__data[0][:, col].reshape(-1) == [unique_x]))
                e = 0
                for unique_y in uniques_y:
                    y = self.__data[1][np.logical_and(y_indexes, (self.__data[1].reshape(-1) == [unique_y]))]
                    if len(y) != 0:
                        e += len(y) / count_x * np.log2(len(y) / count_x)

                res -= (count_x / len(self.__indexes) * e)
        return res

    def disc(self, col):

        res = self.entropy() - self.entropy(col)

        return res

    def __predict(self, data_x):

        if len(self.__branches) == 0:
            vals, counts = np.unique(self.__data[1][self.__indexes], return_counts=True)
            return vals[np.argmax(counts)]
        else:
            corresponding_branch = None
            for branch in self.__branches:
                if branch.get_rule().is_satisfied(data_x):
                    corresponding_branch = branch
                    break
            if corresponding_branch == None:
                # No such data point was detected before
                unique, counts = np.unique(self.__data[1], return_counts=True)
                return unique[np.argmax(counts)]
            return corresponding_branch.__predict(data_x)

    def predict(self, data_x):

        results = np.empty((data_x.shape[0], 1), dtype=self.__data[1].dtype)
        data = data_x.copy()

        data, _ = self.float_to_one_hot(data)

        for i, dp in enumerate(data):
            results[i, 0] = self.__predict(dp.reshape(1, -1))

        return results


def float_to_one_hot(data, cols="all", classes_n=3):
    if cols == "all":
        cols = [col for col in range(data.shape[1])]
    data = data.copy()
    for col in cols:
        data = data[data[:, col].argsort()]
        for cl in range(classes_n):
            data[cl * len(data) // classes_n: (cl + 1) * len(data) // classes_n, col] = cl
    return data


def str_to_one_hot(data, cols=[-1]):
    if cols == "all":
        cols = [col for col in data.columns]
    data = data.copy()
    dictionary = np.unique(data[:, cols])
    for col1 in cols:
        for i, col2 in enumerate(dictionary):
            data[data[:, col1] == col2, col1] = i
    return data, dictionary


def confusion_matrix(y_pred, y_true):
    result = [0, 0, 0, 0]

    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            result[0] += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            result[2] += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            result[1] += 1
        else:
            result[3] += 1

    return np.array(result)


def split_train_test(data, train_size=0.7):
    indexes = np.random.permutation(len(data))

    train = data[indexes[:int(len(data) * train_size)]]
    test = data[indexes[int(len(data) * train_size):]]

    return (train, test)


def precision(y_pred, y_true):
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise AssertionError("y_pred and y_true must be of type numpy.array")

    if y_pred.shape != y_true.shape:
        raise AssertionError("y_pred and y_true must be of same shape")

    tp, fp, fn, tn = confusion_matrix(y_pred, y_true)
    return tp / (tp + fp)


def recall(y_pred, y_true):
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise AssertionError("y_pred and y_true must be of type numpy.array")

    if y_pred.shape != y_true.shape:
        raise AssertionError("y_pred and y_true must be of same shape")

    tp, fp, fn, tn = confusion_matrix(y_pred, y_true)
    return tp / (tp + fn)


def accuracy(y_pred, y_true):
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise AssertionError("y_pred and y_true must be of type numpy.array")

    if y_pred.shape != y_true.shape:
        raise AssertionError("y_pred and y_true must be of same shape")

    tp, fp, fn, tn = confusion_matrix(y_pred, y_true)
    return (tp + tn) / (tp + fp + fn + tn)


def specificity(y_pred, y_true):
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise AssertionError("y_pred and y_true must be of type numpy.array")

    if y_pred.shape != y_true.shape:
        raise AssertionError("y_pred and y_true must be of same shape")

    tp, fp, fn, tn = confusion_matrix(y_pred, y_true)
    return tn / (fp + tn)


def f1_score(y_pred, y_true):
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise AssertionError("y_pred and y_true must be of type numpy.array")

    if y_pred.shape != y_true.shape:
        raise AssertionError("y_pred and y_true must be of same shape")

    tp, fp, fn, tn = confusion_matrix(y_pred, y_true)
    return 2 * tp / (2 * tp + fp + fn)


if __name__ == "__main__":
    data = pd.read_csv("heart_disease_dataset.csv", delimiter=';')
    train, test = split_train_test(data.values, 0.5)

    x_cols = list(range(len(data.columns) - 1))
    train_x = train[:, x_cols]
    test_x = test[:, x_cols]

    y_cols = [-1]
    train_y = train[:, y_cols]
    test_y = test[:, y_cols]

    n_groups = 3
    dt = decision_tree((train_x, train_y), n_groups=n_groups)

    print(f"Training on {len(train_x)} data points, validating on {len(test_x)} data points, n_groups: {n_groups}")

    pred = dt.predict(test_x)

    print(f"Accuracy   : {accuracy(pred, test_y)}")
    print(f"Precision  : {precision(pred, test_y)}")
    print(f"Recall     : {recall(pred, test_y)}")
    print(f"Specificity: {specificity(pred, test_y)}")
    print(f"F1 score   : {f1_score(pred, test_y)}")