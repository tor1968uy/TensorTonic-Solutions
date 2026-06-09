import numpy as np

def vae_decoder(z: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (batch, output_dim) with reconstructed data
    """
    # Perform linear projection: x_hat = z · W + b
    x_hat = np.dot(z, W_dec) + b_dec
    
    return x_hat