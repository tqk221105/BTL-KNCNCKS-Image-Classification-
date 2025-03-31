import numpy as np


def mse(y_true, y_pred):
    # TODO


def mse_derivative(y_true, y_pred):
    # TODO: return the vector dE/dY


def CCE(y_true, y_pred):  # Categorical Cross-entropy
    # TODO
    # Since the labels will be one-hot encode so we need to clip predictions
    # in the range [1e-12, 1 - 1e-12] to prevent log(0)
    # e.g. [0.1, 0.1, 0.0, 0.8] which will result in log(0)
   
    # NOTE: should pay attention to training in batches instead of training in single image


def CCE_derivative(y_true, y_pred):
    # TODO
    return -np.array(y_true) / y_pred  # TODO: check for error, output_gradient (4x4)
