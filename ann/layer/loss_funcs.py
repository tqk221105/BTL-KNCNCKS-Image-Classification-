import numpy as np

def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (np.array(y_pred) - np.array(y_true)) / np.size(y_true)

def CCE(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(np.array(y_true) * np.log(y_pred)) / y_true.shape[0]

def CCE_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.array(y_true) / y_pred
