import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step with epsilon inside the square root.
    """
    # 1. Convert to NumPy arrays
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    G = np.asarray(G, dtype=float)

    # 2. Accumulate squared gradients
    new_G = G + (g ** 2)

    # 3. Update parameters using adaptive learning rate
    # The test case requires epsilon to be added inside the square root
    new_w = w - (lr * g) / np.sqrt(new_G + eps)

    return new_w, new_G