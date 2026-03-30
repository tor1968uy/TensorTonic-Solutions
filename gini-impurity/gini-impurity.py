import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    y_left = np.array(y_left)
    y_right = np.array(y_right)
    
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    # Requisito: Manejar nodos vacíos (convención de impureza 0)
    if n_total == 0:
        return 0.0

    def calculate_single_gini(y):
        n = len(y)
        if n == 0:
            return 0.0
        # Hint 1: Obtener frecuencias de clases
        _, counts = np.unique(y, return_counts=True)
        # Probabilidades al cuadrado: sum((count / n)^2)
        probs_squared = np.sum((counts / n) ** 2)
        return 1.0 - probs_squared

    # Calcular Gini para cada hijo
    gini_left = calculate_single_gini(y_left)
    gini_right = calculate_single_gini(y_right)
    
    # Hint 3: Ponderar por la proporción de muestras en cada hijo
    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    
    return float(weighted_gini)