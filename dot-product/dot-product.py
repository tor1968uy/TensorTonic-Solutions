import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Returns: float - the resulting scalar value.
    """
    # 1. Convert inputs to NumPy arrays
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    
    # 2. Check for dimension mismatch
    if x_arr.shape != y_arr.shape:
        raise ValueError("Input arrays must have the same length.")
    
    # 3. Compute dot product: sum(x_i * y_i)
    # Using np.dot is standard for 1D arrays
    result = np.dot(x_arr, y_arr)
    
    return float(result)