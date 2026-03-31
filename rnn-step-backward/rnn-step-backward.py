import numpy as np

def rnn_step_backward(dh, cache):
    """
    Computa el gradiente para un solo paso de tiempo de una celda RNN.
    
    Args:
        dh: Gradiente de la pérdida respecto al estado oculto actual (h_t).
        cache: Lista con [x_t, h_prev, h_t, W, U, b] guardados en el forward pass.
        
    Returns:
        tuple: (dx_t, dh_prev, dW, dU, db)
    """
    # 1. Desempaquetar el cache
    x_t, h_prev, h_t, W, U, b = cache
    
    # Convertir a arrays de numpy para asegurar operaciones vectorizadas
    dh = np.array(dh)
    h_t = np.array(h_t)
    x_t = np.array(x_t)
    h_prev = np.array(h_prev)
    W = np.array(W)
    U = np.array(U)

    # 2. Gradiente a través de la función de activación tanh
    # La derivada de tanh(z) es (1 - tanh(z)^2)
    # dz representa dL/dz donde z = W*x + U*h_prev + b
    dz = dh * (1 - h_t**2)

    # 3. Gradientes para los parámetros (W, U, b)
    # dW = dz (outer) x_t
    dW = np.outer(dz, x_t)
    
    # dU = dz (outer) h_prev
    dU = np.outer(dz, h_prev)
    
    # db = dz (el gradiente del bias es simplemente el gradiente acumulado dz)
    db = dz

    # 4. Gradientes para las entradas (x_t, h_prev)
    # dx_t = W^T @ dz
    dx_t = W.T @ dz
    
    # dh_prev = U^T @ dz
    dh_prev = U.T @ dz

    return dx_t, dh_prev, dW, dU, db

# --- Validación con el ejemplo ---
if __name__ == "__main__":
    dh_ex = [1, 1]
    # Cache: [x_t, h_prev, h_t, W, U, b]
    cache_ex = [
        [0.5, 0.3], [0.1, 0.2], [0.6, 0.4], 
        [[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [0, 0]
    ]
    
    dx, dhp, dW, dU, db = rnn_step_backward(dh_ex, cache_ex)
    print(f"dx: {np.round(dx, 3)}")
    print(f"dh_prev: {np.round(dhp, 3)}")
    print(f"db: {np.round(db, 3)}")