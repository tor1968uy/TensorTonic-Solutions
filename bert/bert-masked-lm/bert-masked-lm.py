import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied.
    
    Strategy:
    - 80% of the time (replace_probs < 0.8): Replace with [MASK]
    - 10% of the time (0.8 <= replace_probs < 0.9): Replace with random token
    - 10% of the time (replace_probs >= 0.9): Keep original token
    """
    # Create copies to avoid modifying the original inputs
    masked_ids = token_ids.copy()
    labels = np.full_like(token_ids, -100)
    
    # Iterate through the array to apply masking logic where mask_positions is True
    for i in range(token_ids.shape[0]):
        for j in range(token_ids.shape[1]):
            if mask_positions[i, j]:
                # 1. Fill the label with the original token ID
                labels[i, j] = token_ids[i, j]
                
                # 2. Determine the replacement based on the 80-10-10 rule
                prob = replace_probs[i, j]
                
                if prob < 0.8:
                    # 80% replace with [MASK]
                    masked_ids[i, j] = mask_token_id
                elif prob < 0.9:
                    # 10% replace with random token
                    masked_ids[i, j] = random_tokens[i, j]
                else:
                    # 10% keep original (do nothing to masked_ids)
                    pass
                    
    return masked_ids, labels

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # Initialize weights with small random values and bias with zeros
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        Input shape: (batch, seq_len, hidden_size)
        Output shape: (batch, seq_len, vocab_size)
        """
        # Matrix multiplication using the @ operator for cleaner syntax
        logits = hidden_states @ self.W + self.b
        return logits