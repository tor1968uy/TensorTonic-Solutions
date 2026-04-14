import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    
    Formula: gamma * (x - mean) / sqrt(variance + eps) + beta
    """
    
    # 1. Compute mean and variance along the last axis (feature dimension)
    # keepdims=True is essential for proper broadcasting during subtraction/division
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # 2. Normalize the input
    # Subtract the mean and divide by the standard deviation (sqrt of variance + eps)
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    # 3. Scale and shift (using gamma and beta)
    # Since gamma and beta are shape (d_model,), they broadcast across the leading dimensions
    output = gamma * x_normalized + beta
    
    return output