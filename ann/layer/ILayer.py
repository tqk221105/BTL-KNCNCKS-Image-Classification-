


from abc import ABC, abstractmethod


class ILayer:
    def __init__(self):
        
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        pass
