import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Imputación de valores faltantes (NaN) por columna o elemento.
    """
    # 1. Convertimos a array de NumPy y forzamos a float para manejar NaNs
    X = np.array(X, dtype=float)
    
    # Guardamos si la entrada original era 1D para devolver el mismo formato
    is_1d = X.ndim == 1
    
    # Hint 2: Si es 2D, iteramos sobre columnas. Si es 1D, lo tratamos como una sola columna.
    # Convertimos temporalmente a 2D (N, 1) si es 1D para unificar la lógica
    if is_1d:
        X_2d = X.reshape(-1, 1)
    else:
        X_2d = X.copy()
    
    rows, cols = X_2d.shape
    
    for j in range(cols):
        col = X_2d[:, j]
        
        # Hint 1: Encontrar NaNs y valores válidos
        nan_mask = np.isnan(col)
        valid_values = col[np.logical_not(nan_mask)]
        
        # Hint 3: Manejar columnas con puros NaNs
        if valid_values.size > 0:
            if strategy == 'mean':
                stat = np.mean(valid_values)
            elif strategy == 'median':
                stat = np.median(valid_values)
            
            # Rellenar NaNs con el estadístico
            X_2d[nan_mask, j] = stat
        else:
            # Requisito del ejemplo: All-NaN col → 0
            X_2d[nan_mask, j] = 0.0
            
    # Devolvemos al formato original (1D o 2D) como lista o array según prefieras
    result = X_2d.flatten() if is_1d else X_2d
    return result.tolist() if isinstance(X.tolist(), list) else result