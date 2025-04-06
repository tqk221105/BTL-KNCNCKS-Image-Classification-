


import numpy as np
from ann.layer.ILayer import ILayer


class FCLayer(ILayer):
    def __init__(self, X_size, Y_size):
        super().__init__()
        self.weights = np.random.randn(X_size, Y_size) * 0.01
        self.bias = np.zeros((1, Y_size))
        self.input = None

    def forward(self, X):
        self.input = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient
       
