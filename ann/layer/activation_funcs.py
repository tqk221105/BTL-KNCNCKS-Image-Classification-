'''
import numpy as np
from ann.layer.ILayer import ILayer
from ann.layer.activation import Activation


class Softmax(ILayer):
    def forward(self, input):
        
        
    def backward(self, output_gradient, learning_rate):
        
        


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):

        def sigmoid_derivative(x):
            


class ReLU(Activation):
    def __init__(self):
        # TODO
        def relu(x):
            # TODO
            return max(0, x)

        def relu_derivative(x):
            # TODO
            return 1 if x > 0 else 0

        super().__init__(relu, relu_derivative)


class Tanh(Activation):
    def __init__(self):
        # TODO
        def tanh(x):
            return np.tanh(x)

        def tanh_derivative(x):
            return 1 / (np.cosh(x) ** 2)

        super().__init__(tanh, tanh_derivative)
'''


import numpy as np
from ann.layer.ILayer import ILayer
from ann.layer.activation import Activation


class Softmax(ILayer):
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities 
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * self.output * (1 - self.output)
        return input_gradient


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
            return 1 / (np.cosh(x) ** 2)

        super().__init__(tanh, tanh_derivative)



