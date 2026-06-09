import numpy as np

class VAE:
    def __init__(self, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray):
        """
        Initialize VAE with concrete weight matrices.
        """
        # Encoder weights and biases
        self.W_mu = W_mu
        self.b_mu = b_mu
        self.W_logvar = W_logvar
        self.b_logvar = b_logvar
        
        # Decoder weights and biases
        self.W_dec = W_dec
        self.b_dec = b_dec
    
    def forward(self, x: np.ndarray, epsilon: np.ndarray) -> dict:
        """
        Full forward pass: encode -> reparameterize -> decode.
        Returns dict with "recon", "mu", "log_var".
        """
        # 1. Encode: Map input space to latent distribution parameters
        mu = np.dot(x, self.W_mu) + self.b_mu
        log_var = np.dot(x, self.W_logvar) + self.b_logvar
        
        # 2. Reparameterize: z = μ + σ ⊙ ϵ
        std = np.exp(0.5 * log_var)
        z = mu + std * epsilon
        
        # 3. Decode: Map latent space back to reconstruction space
        recon = self.generate(z)
        
        return {
            "recon": recon,
            "mu": mu,
            "log_var": log_var
        }
    
    def generate(self, z: np.ndarray) -> np.ndarray:
        """
        Generate samples from given latent vectors.
        """
        # Map latent vector z back to data space
        return np.dot(z, self.W_dec) + self.b_dec