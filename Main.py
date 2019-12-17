import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from UFAZ.AIproject.functions import Functions as nn

# import csv file
heart_disease = pd.read_csv("heart_disease_dataset.csv", delimiter=';')
# heart_disease.head()


# get inputs (columns - list of column names, x - the inputs)
columns = pd.read_csv('heart_disease_dataset.csv', delimiter=';', nrows=1).columns.tolist()
x = pd.DataFrame(heart_disease, columns=columns)
x = nn.normalize(x.values)

# Get Output, flatten and encode to one-hot
output_column = ['target']
y = pd.DataFrame(heart_disease, columns=output_column)
y = y.values  # convert to matrix
y = y.flatten()  # collapses matrix into 1D array
y = nn.to_one_hot(y)  # converting to one-hot array

# Split data to training and validation data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# Weights
w0 = 2 * np.random.random((14, 5)) - 1  # for input   - 14 inputs, 3 outputs
w1 = 2 * np.random.random((5, 2)) - 1  # for layer 1 - 5 inputs, 3 outputs
# So from weights initialization we get that we have 5 inputs in our hidden layer

# learning rate
lr = 0.05

# Errors - it will be needed to draw a graph
errors = []

# Training process
for i in range(100000):
    # Feed forward
    input_layer = X_train
    hidden_layer = nn.sigmoid(np.dot(input_layer, w0))
    output_layer = nn.sigmoid(np.dot(hidden_layer, w1))

    # Back propagation using gradient descent
    layer2_error = y_train - output_layer
    layer2_delta = layer2_error * nn.sigmoid_deriv(output_layer)

    layer1_error = layer2_delta.dot(w1.T)
    layer1_delta = layer1_error * nn.sigmoid_deriv(hidden_layer)

    # updating weights using previous results multiplying it with learning rate
    w1 += hidden_layer.T.dot(layer2_delta) * lr
    w0 += input_layer.T.dot(layer1_delta) * lr

    error = np.mean(np.abs(layer2_error))
    errors.append(error)
    accuracy = (1 - error) * 100
    if i % 10000 == 0:
        print('Cost after {} iterations: {}'.format(i, errors[i]))
        print('Accuracy after {} iterations: {}'.format(i, accuracy))

# Plot the accuracy chart using errors
plt.plot(errors)
plt.xlabel('Training')
plt.ylabel('Error')
plt.show()

print("Training Accuracy: {}".format(str(round(accuracy, 2)) + "%"))

input_layer_test = X_test
hidden_layer_test = nn.sigmoid(np.dot(input_layer_test, w0))
output_layer_test = nn.sigmoid(np.dot(hidden_layer_test, w1))

layer2_error = y_test - output_layer_test

error = np.mean(np.abs(layer2_error))
accuracy = (1 - error) * 100

print("Validation Accuracy: {}".format(str(round(accuracy, 2)) + "%"))
