import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    n_samples, n_features = X.shape
    
    # 1. Calcular X^T * X
    xtx = X.T @ X
    
    # 2. Crear la matriz identidad y escalarla por lambda (Hint 1)
    # I = [[1 if i==j else 0 for j in range(d)] for i in range(d)]
    I = np.eye(n_features)
    ridge_term = lam * I
    
    # 3. Sumar lambda * I a X^T * X
    regularized_matrix = xtx + ridge_term
    
    # 4. Invertir la matriz regularizada
    # Usamos np.linalg.inv como pide el requerimiento
    matrix_inv = np.linalg.inv(regularized_matrix)
    
    # 5. Multiplicar por X^T * y
    # w = (X^T X + lambda*I)^-1 * X^T * y
    weights = matrix_inv @ X.T @ y
    
    # Retornar como lista de floats
    return weights.tolist()