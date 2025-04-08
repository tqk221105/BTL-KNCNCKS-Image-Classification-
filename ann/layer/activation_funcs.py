import numpy as np
from ann.layer.ILayer import ILayer
from ann.layer.activation import Activation

class Softmax(ILayer):
    def __init__(self):
        self.output = None

    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_derivative)


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_derivative(x):
            return (x > 0).astype(float)

        super().__init__(relu, relu_derivative)


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_derivative(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_derivative)
