import numpy as np

def relu(x):
    return np.maximum(0, x)

def identity_block(x, W1, W2):
    # Convert inputs to numpy arrays to enable .T and matrix operations
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    
    # 1. Save identity for the skip connection
    identity = x
    
    # 2. First linear layer: ReLU(x @ W1.T)
    # Using @ or np.dot is fine; .T requires a numpy array
    h = relu(x @ W1.T)
    
    # 3. Second linear layer: h @ W2.T
    f_x = h @ W2.T
    
    # 4. Add identity and final ReLU
    output = relu(f_x + identity)
    
    return output