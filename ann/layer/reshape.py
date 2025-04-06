'''
import numpy as np
from ann.layer.ILayer import ILayer


class Reshape(ILayer):
    def __init__(self, input_shape, output_shape):
        

    def forward(self, X):

    def backward(self, output_gradient, learning_rate):
'''

import numpy as np
from ann.layer.ILayer import ILayer

class Reshape(ILayer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, X):
        self.input_data = X
        return X.reshape(self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)
