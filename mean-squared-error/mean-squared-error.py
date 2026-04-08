import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Calcula el Error Cuadrático Medio (MSE).
    
    Args:
        y_pred: Predicciones del modelo.
        y_true: Valores reales (targets).
        
    Returns:
        float: El valor del MSE, o None si las dimensiones no coinciden.
    """
    # 1. Convertir entradas a arreglos de NumPy
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # 2. Verificar que las dimensiones coincidan
    if y_pred.shape != y_true.shape:
        return None
    
    # 3. Calcular el MSE
    # (pred - true)**2 calcula la diferencia al cuadrado elemento por elemento
    # np.mean() suma esos valores y divide por N
    mse = np.mean((y_pred - y_true)**2)
    
    return float(mse)