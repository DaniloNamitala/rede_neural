import math
import numpy as np

def relu(x):
    return max(0, x)

def relu_derivative(x):
    if x > 0:
        return 1
    return 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neurons a1 to a5 represented by indexes 0 to 4
v = [0, 0, 0, 0, 0]
e = [0, 0, 0, 0, 0]
w = np.array([
    [0.,  0.,  0.,   0., 0.],
    [3.,  0.,  0.,   0., 0.],
    [-4.,  1.,  0.,   0., 0.],
    [-1., -3.,  0.,   0., 0.], 
    [0.,  0.,  2., -10., 0.], 
])
b = np.array([0., 0., 0., 0., 0.])  # Bias terms
f = [sigmoid, relu, sigmoid, sigmoid, None]
fd = [sigmoid_derivative, relu_derivative, sigmoid_derivative, sigmoid_derivative, None]

def forward(entry):
    v[-1] = entry
    for i in range(len(v) - 2, -1, -1):
        v[i] = 0
        for j in range(len(v)):
            v[i] += v[j] * w[j][i]
        v[i] += b[i]  # Add bias term
        v[i] = f[i](v[i])

def calculate_error(expected):
    global e
    e = [0, 0, 0, 0, 0]
    e[0] = (v[0] - expected) * fd[0](v[0])

    for i in range(1, len(v) - 1):
        for j in range(len(v)):
            e[i] += e[j] * w[j][i]
        e[i] = e[i] * fd[i](v[i])

def update_weights():
    for i in range(len(w)):
        for j in range(len(w[i])):
            w[j][i] = w[j][i] - 0.1 * e[i] * v[j]
        b[i] = b[i] - 0.1 * e[i]  # Update bias term

def train(entries, outs):
    for i in range(2*9999):
        for (en, ex) in zip(entries, outs):
            forward(en)
            calculate_error(ex)
            update_weights()

def predict(entry):
    forward(entry)
    print(v[0])

if __name__ == "__main__":
  entries = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
  outs = [0.73212, 0.7339, 0.7838, 0.8903, 0.9820, 0.8114, 0.5937, 0.5219, 0.5049, 0.5002]
  # Call the train function to train the network
  train(entries, outs)

  # Call the predict function with a specific entry
  predict(0.5)
  predict(0.0)
  predict(-3.0)
  predict(3.0)