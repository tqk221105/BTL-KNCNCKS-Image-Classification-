import numpy as np
from ann.layer.ILayer import ILayer

class Reshape(ILayer):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input = None

    def forward(self, X):
        self.input = X
        batch_size = X.shape[0]
        return X.reshape((batch_size, *self.output_shape))

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]
        return output_gradient.reshape((batch_size, *self.input_shape))
