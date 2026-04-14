import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    
    Formula: FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    # 1. First Linear Layer (Expansion)
    # Projects from d_model to d_ff
    # Shape: (batch, seq, d_model) -> (batch, seq, d_ff)
    hidden = np.dot(x, W1) + b1
    
    # 2. ReLU Activation
    # Element-wise non-linearity: max(0, x)
    relu_out = np.maximum(0, hidden)
    
    # 3. Second Linear Layer (Contraction)
    # Projects from d_ff back to d_model
    # Shape: (batch, seq, d_ff) -> (batch, seq, d_model)
    output = np.dot(relu_out, W2) + b2
    
    return output