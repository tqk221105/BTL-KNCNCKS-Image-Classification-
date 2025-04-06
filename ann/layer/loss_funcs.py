'''
import numpy as np


def mse(y_true, y_pred):


def mse_derivative(y_true, y_pred):


def CCE(y_true, y_pred):  


def CCE_derivative(y_true, y_pred):
    return -np.array(y_true) / y_pred  
'''

import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def CCE(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
def CCE_derivative(y_true, y_pred):
    return -np.array(y_true) / y_pred
