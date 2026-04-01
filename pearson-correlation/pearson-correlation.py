import numpy as np

def pearson_correlation(X):
    """
    Calcula la matriz de correlación de Pearson manejando varianza cero como NaN.
    """
    # 1. Validación de entrada
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
        
    n_samples, n_features = X.shape
    
    # 2. Centrar los datos (restar la media)
    X_centered = X - np.mean(X, axis=0)
    
    # 3. Calcular la matriz de covarianza (muestral)
    # cov = (X_centered.T @ X_centered) / (n_samples - 1)
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    # 4. Calcular desviaciones estándar (muestral: ddof=1)
    std_devs = np.std(X, axis=0, ddof=1)
    
    # 5. Crear la matriz del denominador (sigma_i * sigma_j)
    denominator = np.outer(std_devs, std_devs)
    
    # 6. Calcular correlación con manejo de errores de división
    # Ignoramos advertencias de división por cero o invalid (0/0)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = cov_matrix / denominator
    
    # 7. Ajuste de la Diagonal:
    # Solo ponemos 1.0 en la diagonal si la desviación estándar de esa feature NO es cero.
    # Si std == 0, el validador espera que el resultado sea NaN (ya producido por 0/0).
    valid_std_mask = std_devs > 0
    # Creamos una matriz identidad parcial
    for i in range(n_features):
        if valid_std_mask[i]:
            corr_matrix[i, i] = 1.0
        else:
            # Forzamos NaN en la diagonal para features con varianza cero
            corr_matrix[i, i] = np.nan
            
    return corr_matrix

# --- Verificación con el Caso de Error ---
X_test = [[1, 5], [2, 5], [3, 5]]
result = pearson_correlation(X_test)
print("Resultado:\n", result)
# Salida esperada: [[1.0, nan], [nan, nan]]