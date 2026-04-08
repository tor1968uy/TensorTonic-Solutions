import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Returns: float
    """
    # 1. Convert inputs to NumPy arrays for vectorized arithmetic
    x = np.asarray(x)
    y = np.asarray(y)
    
    # 2. Formula: Sum of |x_i - y_i|
    # np.abs() computes the absolute difference element-wise
    # np.sum() aggregates the differences into a single scalar
    distance = np.sum(np.abs(x - y))
    
    return float(distance)