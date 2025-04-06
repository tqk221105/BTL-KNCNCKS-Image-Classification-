'''
import numpy as np
from scipy import signal
from ann.layer.ILayer import ILayer



class Convolutional(ILayer):
    def __init__(self, input_shape, kernel_size, depth):
       
        
                   
    def forward(self, input):
        

    def backward(self, output_gradient, learning_rate):
'''
import numpy as np
from scipy import signal
from ann.layer.ILayer import ILayer

class Convolutional(ILayer):
    def __init__(self, input_shape, kernel_size, depth, stride=1, padding=0):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride
        self.padding = padding
        self.kernels = np.random.randn(depth, kernel_size[0], kernel_size[1], input_shape[2]) * 0.1
        self.bias = np.zeros((depth, 1))

    def forward(self, input):
        self.input = input
        input_height, input_width, input_channels = input.shape
        output_height = (input_height - self.kernel_size[0] + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size[1] + 2 * self.padding) // self.stride + 1
        output = np.zeros((output_height, output_width, self.depth))
        if self.padding > 0:
            input = np.pad(input, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        # Thực hiện convolution
        for d in range(self.depth):
            for i in range(output_height):
                for j in range(output_width):
                    vert_start = i * self.stride
                    vert_end = vert_start + self.kernel_size[0]
                    horiz_start = j * self.stride
                    horiz_end = horiz_start + self.kernel_size[1]
                    patch = input[vert_start:vert_end, horiz_start:horiz_end, :]
                    output[i, j, d] = np.sum(patch * self.kernels[d]) + self.bias[d]

        return output

    def backward(self, output_gradient, learning_rate):
        input_height, input_width, input_channels = self.input.shape
        output_height, output_width, _ = output_gradient.shape
        kernel_gradient = np.zeros_like(self.kernels)
        bias_gradient = np.zeros_like(self.bias)
        input_gradient = np.zeros_like(self.input)
        for d in range(self.depth):
            for i in range(output_height):
                for j in range(output_width):
                    vert_start = i * self.stride
                    vert_end = vert_start + self.kernel_size[0]
                    horiz_start = j * self.stride
                    horiz_end = horiz_start + self.kernel_size[1]

                    patch = self.input[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Cập nhật gradient của bộ lọc và bias
                    kernel_gradient[d] += patch * output_gradient[i, j, d]
                    bias_gradient[d] += output_gradient[i, j, d]

                    # Tính gradient của đầu vào
                    input_gradient[vert_start:vert_end, horiz_start:horiz_end, :] += self.kernels[d] * output_gradient[i, j, d]

        # Cập nhật tham số
        self.kernels -= learning_rate * kernel_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient





