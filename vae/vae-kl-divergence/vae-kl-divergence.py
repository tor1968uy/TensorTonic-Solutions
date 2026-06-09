import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns: float scalar KL divergence averaged over the batch
    """
    # 1. Compute the KL divergence for each element in the batch and latent dimensions
    # Formula components: 1 + log(σ²) - μ² - σ²
    kl_elementwise = 1.0 + log_var - np.square(mu) - np.exp(log_var)
    
    # 2. Sum over the latent dimensions (axis=1) for each sample in the batch
    kl_per_sample = -0.5 * np.sum(kl_elementwise, axis=1)
    
    # 3. Average the total KL divergence over the batch (axis=0)
    kl_loss = np.mean(kl_per_sample)
    
    return float(kl_loss)