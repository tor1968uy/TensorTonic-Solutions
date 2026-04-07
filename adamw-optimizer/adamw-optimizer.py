import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    Returns: (new_w, new_m, new_v) as NumPy arrays.
    """
    # 1. Convert all inputs to NumPy arrays for vectorized math
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    g = np.asarray(grad, dtype=float)

    # 2. Update First Moment (Exponential Moving Average of gradients)
    # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    new_m = beta1 * m + (1 - beta1) * g

    # 3. Update Second Moment (Exponential Moving Average of squared gradients)
    # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    new_v = beta2 * v + (1 - beta2) * (g ** 2)

    # 4. Apply Decoupled Weight Decay
    # Part of the update is based strictly on the current weight value
    w_decayed = w - lr * weight_decay * w

    # 5. Apply Adaptive Gradient Update
    # Final weights = decayed weights - lr * (m / (sqrt(v) + eps))
    new_w = w_decayed - lr * (new_m / (np.sqrt(new_v) + eps))

    return new_w, new_m, new_v