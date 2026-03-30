import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Convertir a arrays de numpy por seguridad
    X = np.array(X)
    y = np.array(y)
    
    # 1. Calcular el producto de la transpuesta por X: (X^T * X)
    xtx = X.T @ X
    
    # 2. Calcular la inversa de (X^T * X)
    # Nota: np.linalg.inv es estándar, pero solve() suele ser más estable numéricamente
    xtx_inv = np.linalg.inv(xtx)
    
    # 3. Multiplicar por X^T y luego por y: (X^T * X)^-1 * X^T * y
    w = xtx_inv @ X.T @ y
    
    return w