import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Escala X al rango [0,1]. Por defecto escala por columna (axis=0).
    """
    X = np.asanyarray(X, dtype=float)
    
    # 1. Calcular el mínimo y máximo a lo largo del eje especificado
    # keepdims=True es crucial para que la resta y división se alineen (broadcasting)
    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)
    
    # 2. Calcular el denominador (rango)
    denominator = x_max - x_min
    
    # 3. Aplicar eps para evitar la división por cero si max == min
    # np.maximum comparará cada elemento del denominador con eps
    denominator = np.maximum(denominator, eps)
    
    # 4. Aplicar la fórmula: (x - min) / (max - min)
    X_scaled = (X - x_min) / denominator
    
    return X_scaled