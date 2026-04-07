import numpy as np

def expected_value_discrete(x, p):
    """
    Compute the expected value of a discrete random variable.
    Returns: float expected value.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    
    # 1. Shape validation
    if x.shape != p.shape:
        raise ValueError("Input arrays x and p must have the same shape.")
        
    # 2. Probability sum validation
    # We use np.allclose to handle floating-point precision issues
    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1.0.")
        
    # 3. Calculate E[X] = sum(x_i * p_i)
    # Using np.dot or np.sum(x * p) are both efficient vectorized approaches
    expected_val = np.sum(x * p)
    
    return float(expected_val)