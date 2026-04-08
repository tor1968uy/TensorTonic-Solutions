import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    Returns: A NumPy array of the same shape as input x.
    """
    # 1. Ensure input is a NumPy array for vectorized operations
    x_arr = np.asarray(x)
    
    # 2. Apply the piecewise formula:
    # If x >= 0: return x
    # If x < 0: return alpha * x
    result = np.where(x_arr >= 0, x_arr, alpha * x_arr)
    
    return result