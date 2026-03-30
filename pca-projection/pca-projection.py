import numpy as np

def pca_projection(X, k):
    X = np.array(X, dtype=float)
    n, d = X.shape
    
    means = np.mean(X, axis=0)
    X_centered = X - means
    
    # SVD de X_centered directamente (más estable que eigendecomposición)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Vt tiene shape (min(n,d), d), cada fila es un componente principal
    W = Vt[:k].T  # shape (d, k)
    
    # Normalización de signo: primer elemento no nulo de cada columna debe ser positivo
    for j in range(W.shape[1]):
        for val in W[:, j]:
            if abs(val) > 1e-9:
                if val < 0:
                    W[:, j] = -W[:, j]
                break
    
    projection = X_centered @ W
    projection[np.abs(projection) < 1e-10] = 0.0
    
    return [row.tolist() for row in projection]