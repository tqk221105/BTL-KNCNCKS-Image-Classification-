"""
    w_ij
x_i -----> y_j = w_1j.x_1 + w_2j.x_2 +...+ w_ij.x_i + b_j


    y_1        w_11  w_12  ...  w_1i         x_1           b_1
    y_2        w_21  w_22  ...  w_2i         x_2           b_2
    .     =   .    .    .    .     x     .      +     .
    .         .    .     .   .           .            .
    .         .    .      .  .           .            .
    y_j        w_j1  w_j2  ...  w_ji         x_i           b_j

    Y = W x X + B

Y(j x 1)
W(j x i)
X(i x 1)
B(j x 1)
"""


import numpy as np
from ann.layer.ILayer import ILayer


class FCLayer(ILayer):
    def __init__(self, X_size, Y_size):
        super().__init__()
        

    def forward(self, X):
        # TODO: return output Y
        

    def backward(self, output_gradient, learning_rate):
                        # dE/dY(j x 1)
        # TODO: update Weight and Bias, then return input gradient X_gradient
       
