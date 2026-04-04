import numpy as np

def log_transform(series):
    return np.log1p(series)

def inverse_log(series):
    return np.expm1(series)