import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    # 1. Compute Reconstruction Loss (Sum squared differences over features, then average over batch)
    squared_errors = np.square(x - x_recon)
    recon_per_sample = np.sum(squared_errors, axis=1)
    recon_loss = np.mean(recon_per_sample)
    
    # 2. Compute KL Divergence Loss (Sum over latent dimensions, then average over batch)
    kl_elementwise = 1.0 + log_var - np.square(mu) - np.exp(log_var)
    kl_per_sample = -0.5 * np.sum(kl_elementwise, axis=1)
    kl_loss = np.mean(kl_per_sample)
    
    # 3. Sum both components to get the total loss
    total_loss = recon_loss + kl_loss
    
    # Return values explicitly cast to native Python floats
    return {
        "total": float(total_loss),
        "recon": float(recon_loss),
        "kl": float(kl_loss)
    }