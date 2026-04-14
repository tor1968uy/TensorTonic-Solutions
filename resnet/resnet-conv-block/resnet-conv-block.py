import numpy as np

def relu(x):
    return np.maximum(0, x)

def conv_block(x, W1, W2, Ws):
    # Ensure inputs are numpy arrays
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)
    
    # 1. Main Path
    # First linear + ReLU
    h = relu(x @ W1)
    # Second linear (No ReLU here!)
    z = h @ W2
    
    # 2. Shortcut Path
    s = x @ Ws
    
    # 3. Final Output (Sum + ReLU)
    # Adding the original signal (s) to the residual (z)
    output = relu(z + s)
    
    return output