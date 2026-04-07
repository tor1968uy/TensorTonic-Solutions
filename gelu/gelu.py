import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Convert input to a numpy array of floats
    x = np.asarray(x, dtype=float)
    
    # Vectorize the math.erf function to handle arrays
    vec_erf = np.vectorize(math.erf)
    
    # Apply the GELU formula
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5 * x * (1 + vec_erf(x / np.sqrt(2)))