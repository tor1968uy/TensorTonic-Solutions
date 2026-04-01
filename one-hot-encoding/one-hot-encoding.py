import numpy as np

def one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=int)
    if num_classes is None:
        num_classes = y.max() + 1
    result = np.zeros((len(y), num_classes), dtype=float)
    result[np.arange(len(y)), y] = 1.0
    return result