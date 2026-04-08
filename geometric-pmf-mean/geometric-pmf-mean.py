import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    Returns: (pmf_array, mean)
    """
    # 1. Convert input k to a NumPy array for vectorized math
    k_arr = np.array(k)
    
    # 2. Compute PMF: P(X=k) = (1-p)^(k-1) * p
    # This represents k-1 failures followed by 1 success
    pmf = ((1 - p) ** (k_arr - 1)) * p
    
    # 3. Compute Theoretical Mean: E[X] = 1/p
    mean = 1.0 / p
    
    return pmf, float(mean)