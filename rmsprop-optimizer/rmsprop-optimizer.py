import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Hint 1: Convertir entradas a NumPy arrays para evitar el error de tipos
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    # Hint 1 & 2: Actualizar s usando el promedio móvil exponencial (EMA)
    # s_new = beta * s + (1 - beta) * g^2
    s_new = beta * s + (1 - beta) * (g * g)

    # Actualizar el peso w
    # w_new = w - (lr / sqrt(s_new + eps)) * g
    w_new = w - (lr * g) / (np.sqrt(s_new) + eps)

    return w_new, s_new