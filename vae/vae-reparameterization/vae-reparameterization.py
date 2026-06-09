import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray z of shape (batch, latent_dim) sampled via reparameterization
    """
    # Convert log-variance to standard deviation: σ = exp(0.5 * log_var)
    std = np.exp(0.5 * log_var)
    
    # Apply the deterministic transformation: z = μ + σ ⊙ ϵ
    z = mu + std * epsilon
    
    return z
