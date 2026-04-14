import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Formula per layer: out = x + (x @ Jacobian_F)
    Or equivalently: out = x @ (I + Jacobian_F)
    """
    # Start with the initial signal vector
    current_signal = x.astype(float)
    
    for grad_F in gradients_F:
        # With skip connection: the signal passes through the identity path 
        # AND the transformation path, then they sum up.
        # current_signal @ grad_F gives the transformed signal
        current_signal = current_signal + np.dot(current_signal, grad_F)
        
    return current_signal

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    Formula per layer: out = x @ Jacobian_F
    """
    current_signal = x.astype(float)
    
    for grad_F in gradients_F:
        # Without skip connection: the signal MUST pass through the transformation.
        # If the Jacobian values are small (< 1), the signal shrinks every layer.
        current_signal = np.dot(current_signal, grad_F)
        
    return current_signal