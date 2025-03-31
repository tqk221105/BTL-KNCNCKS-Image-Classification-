import numpy as np
from scipy import signal
from ann.layer.ILayer import ILayer


"""
Y: output
I: input
K: kernel
shape(Y) = shape(I) - shape(K) + 1
"""


class Convolutional(ILayer):
    def __init__(self, input_shape, kernel_size, depth):
        # input_shape: width x height x input channels (input channels, height, width)
        # kernel_size: size of kernel (square matrix kernel_size x kernel_size)
        # depth: number of kernels or number of output channels.
        
        
                    # input channels
        
        # kernel_size x kernel_size x input channels x output channels

        # number of kernels x 1 x 1
        # each kernel has a unique bias value.

    def forward(self, input):
        # input: width x height x input channels
        # input.shape: (input channels, height, width)

        # height and width of input image
        

        # height and width of output image
        

    def backward(self, output_gradient, learning_rate):
        # output_gradient: out_width x out_height x depth (depth, out_height, out_width)
        # update self.bias

        
        # update self.filter & input_gradient
       
                                            # convolve2d? We have to rotate the filter 180deg
                                            # dE/dX_j = sum(dE/dY_i *full rot(K)) (for i in range [1, input channels]
                                            #         = sum(dE/dY_i *convolve K)




