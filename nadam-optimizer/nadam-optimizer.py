import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    w    = np.asarray(w,    dtype=float)
    m    = np.asarray(m,    dtype=float)
    v    = np.asarray(v,    dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Actualizar momentos (sin bias correction)
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad**2

    # Nesterov lookahead: combina m_new con el gradiente actual
    m_nadam = beta1 * m_new + (1 - beta1) * grad

    w_new = w - lr * m_nadam / (np.sqrt(v_new) + eps)

    return w_new, m_new, v_new