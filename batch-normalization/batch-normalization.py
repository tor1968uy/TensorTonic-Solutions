import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Realiza la normalización por lotes (Batch Normalization) para (N, D) o (N, C, H, W).
    """
    x = np.asanyarray(x, dtype=np.float64)
    gamma = np.asanyarray(gamma, dtype=np.float64)
    beta = np.asanyarray(beta, dtype=np.float64)
    
    ndim = x.ndim
    
    # 1. Determinar los ejes de reducción y preparar el broadcasting de gamma/beta
    if ndim == 2:
        # Caso (N, D): Normalizar sobre el eje 0 (batch)
        axis = (0,)
        # Gamma y Beta ya tienen forma (D,), el broadcasting funciona directo
    elif ndim == 4:
        # Caso (N, C, H, W): Normalizar sobre (0, 2, 3) (batch y espacial)
        axis = (0, 2, 3)
        # Reshape gamma y beta a (1, C, 1, 1) para aplicar a cada canal
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    else:
        raise ValueError("Input debe ser 2D o 4D.")

    # 2. Calcular media y varianza con keepdims=True para facilitar la división
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)

    # 3. Normalizar
    x_hat = (x - mean) / np.sqrt(var + eps)

    # 4. Escalar y desplazar (Scale and Shift)
    out = gamma * x_hat + beta

    return out

# --- Validación con ejemplos ---
# Ejemplo 2D
x2d = [[1, 2], [3, 6], [5, 10]]
g2d, b2d = [1, 0.5], [0, 1]
print("BN 2D Output:\n", batch_norm_forward(x2d, g2d, b2d))

# Ejemplo 4D
x4d = [[[[1]],[[2]]], [[[3]],[[4]]]] # (2, 2, 1, 1)
g4d, b4d = [1, 0.5], [0, -1]
print("\nBN 4D Output:\n", batch_norm_forward(x4d, g4d, b4d))