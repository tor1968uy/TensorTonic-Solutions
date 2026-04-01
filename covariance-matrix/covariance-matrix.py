import numpy as np

def covariance_matrix(X):
    """
    Calcula la matriz de covarianza muestral de un conjunto de datos X.
    """
    # 1. Validación de entrada
    X = np.asarray(X, dtype=float)
    
    # Debe ser 2D y tener al menos 2 muestras para calcular covarianza muestral
    if X.ndim != 2 or X.shape[0] < 2:
        return None
        
    n_samples, n_features = X.shape
    
    # 2. Centrar los datos (restar la media de cada característica)
    # np.mean con axis=0 calcula la media de cada columna (D medias)
    X_centered = X - np.mean(X, axis=0)
    
    # 3. Calcular el producto matricial de los datos centrados
    # Usamos la transpuesta de X_centered para obtener una matriz de (D, D)
    # (D, N) @ (N, D) -> (D, D)
    cov = np.dot(X_centered.T, X_centered)
    
    # 4. Dividir por N-1 (estimador insesgado de la covarianza muestral)
    return cov / (n_samples - 1)

# --- Verificación con ejemplos ---
# Ejemplo 1: Tendencia lineal perfecta
X1 = [[1, 2], [2, 3], [3, 4]]
print("Ejemplo 1:\n", covariance_matrix(X1))

# Ejemplo 2: Correlación negativa
X2 = [[1, 0], [0, 1]]
print("\nEjemplo 2:\n", covariance_matrix(X2))

# Ejemplo 3: Caso inválido (1 muestra)
X3 = [[1, 2, 3]]
print("\nEjemplo 3 (N=1):", covariance_matrix(X3))