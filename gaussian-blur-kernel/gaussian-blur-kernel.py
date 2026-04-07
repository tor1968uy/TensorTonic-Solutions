import numpy as np
import math

def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    Returns a 2D list of floats.
    """
    # 1. Initialize a grid of zeros
    kernel = np.zeros((size, size))
    center = size // 2
    
    # 2. Compute raw Gaussian weights
    # Formula: G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
    sum_val = 0.0
    for i in range(size):
        for j in range(size):
            # Calculate offsets from the center
            y = i - center
            x = j - center
            
            # Compute the unnormalized exponent
            exponent = -(x**2 + y**2) / (2.0 * sigma**2)
            weight = math.exp(exponent)
            
            kernel[i, j] = weight
            sum_val += weight
            
    # 3. Normalize the kernel so it sums to 1.0
    kernel = kernel / sum_val
    
    return kernel.tolist()