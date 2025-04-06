import numpy as np
from ann.layer.ILayer import ILayer


class Activation(ILayer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input 
        return self.activation(input)

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * self.activation_derivative(self.input)
        return input_gradient
