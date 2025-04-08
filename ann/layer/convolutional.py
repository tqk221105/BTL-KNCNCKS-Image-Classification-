import numpy as np
from scipy import signal
from ann.layer.ILayer import ILayer


class Convolutional(ILayer):
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.depth = depth

        c, h, w = input_shape
        kh, kw = kernel_size
        self.kernels = np.random.randn(depth, c, kh, kw) * np.sqrt(2. / (c * kh * kw))
        self.biases = np.zeros((depth, 1))

    def forward(self, input):
        self.input = input
        c, h, w = self.input_shape
        kh, kw = self.kernel_size
        batch_size = input.shape[0]

        output_height = h - kh + 1
        output_width = w - kw + 1
        output = np.zeros((batch_size, self.depth, output_height, output_width))

        for i in range(batch_size):
            for d in range(self.depth):
                for ch in range(c):
                    output[i, d] += signal.correlate2d(input[i, ch], self.kernels[d, ch], mode='valid')
                output[i, d] += self.biases[d]
        return output

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]
        c, h, w = self.input_shape
        kh, kw = self.kernel_size

        input_gradient = np.zeros_like(self.input)
        kernels_gradient = np.zeros_like(self.kernels)
        biases_gradient = np.zeros_like(self.biases)

        for i in range(batch_size):
            for d in range(self.depth):
                for ch in range(c):
                    input_gradient[i, ch] += signal.convolve2d(
                        output_gradient[i, d], self.kernels[d, ch], mode='full')

                    # Gradient với kernel
                    kernels_gradient[d, ch] += signal.correlate2d(
                        self.input[i, ch], output_gradient[i, d], mode='valid')

                biases_gradient[d] += np.sum(output_gradient[i, d])

        self.kernels -= learning_rate * kernels_gradient / batch_size
        self.biases -= learning_rate * biases_gradient / batch_size

        return input_gradient
