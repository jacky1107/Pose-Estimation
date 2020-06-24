import numpy as np

def stdDev(data):
    r, c = data.shape
    mean = np.mean(data, axis=0)
    differ_vector = data - mean
    dist = differ_vector[:, 0] ** 2 + differ_vector[:, 1] ** 2
    return np.sqrt(np.sum(dist) / r)
