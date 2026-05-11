import numpy as np

def vae_encoder(x: np.ndarray, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray) -> dict:
    """
    VAE Encoder: maps input x to latent distribution parameters (mu, log_var).
    
    Args:
        x: Input data of shape (N, D)
        W_mu: Weight matrix for mean, shape (D, latent_dim)
        b_mu: Bias for mean, shape (latent_dim,)
        W_logvar: Weight matrix for log variance, shape (D, latent_dim)
        b_logvar: Bias for log variance, shape (latent_dim,)
    
    Returns:
        dict with 'mu' and 'log_var' as np.ndarrays of shape (batch_size, latent_dim)
    """
    mu = np.dot(x, W_mu) + b_mu
    log_var = np.dot(x, W_logvar) + b_logvar
    
    return {"mu": mu, "log_var": log_var}
