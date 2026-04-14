import numpy as np

def relu(x):
    """Standard ReLU activation function."""
    return np.maximum(0, x)

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Implements the Bottleneck Block forward pass.
    Pattern: ReLU(W3 * ReLU(W2 * ReLU(W1 * x))) + shortcut
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    
    # 1. Main Path
    # W1: Reduce (in_channels -> bottleneck_channels)
    h1 = relu(x @ W1)
    
    # W2: Process (bottleneck_channels -> bottleneck_channels)
    h2 = relu(h1 @ W2)
    
    # W3: Expand (bottleneck_channels -> out_channels)
    # Note: Traditionally, the third layer doesn't have a ReLU 
    # until AFTER the addition with the shortcut.
    z = h2 @ W3
    
    # 2. Shortcut Path
    # Use learned projection if Ws is provided, otherwise use identity
    if Ws is not None:
        shortcut = x @ np.array(Ws)
    else:
        shortcut = x
        
    # 3. Final Addition and Activation
    # The skip connection preserves the gradient flow
    output = relu(z + shortcut)
    
    return output