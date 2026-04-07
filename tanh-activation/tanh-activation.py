import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    Supports scalars, lists, and NumPy arrays.
    """
    # 1. Convert input to a NumPy array for vectorized math
    x = np.asarray(x, dtype=float)
    
    # 2. Compute e^x and e^-x
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    
    # 3. Apply the Tanh formula: (e^x - e^-x) / (e^x + e^-x)
    # Note: NumPy also provides a built-in np.tanh(x) for production
    result = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    
    # 4. Handle scalar inputs to return a 1D array as per example requirements
    if x.ndim == 0:
        return np.array([result])
        
    return result