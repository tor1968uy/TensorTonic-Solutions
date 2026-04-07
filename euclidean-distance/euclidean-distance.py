import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Returns: float - the resulting scalar distance.
    """
    # 1. Ensure inputs are NumPy arrays for vectorized operations
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    
    # 2. Compute the element-wise difference
    diff = x_arr - y_arr
    
    # 3. Calculate the L2 Norm: sqrt(sum of squared differences)
    # Using np.linalg.norm is the most efficient and standard way
    dist = np.linalg.norm(diff)
    
    return float(dist)