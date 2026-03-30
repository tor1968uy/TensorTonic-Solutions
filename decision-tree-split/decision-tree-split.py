import numpy as np

def calculate_gini(y):
    """Calcula la impureza de Gini: 1 - sum(p_i^2)"""
    m = len(y)
    if m == 0:
        return 0
    counts = np.bincount(y)
    probs = counts / m
    return 1 - np.sum(probs**2)

def decision_tree_split(X, y):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    n_samples, n_features = X.shape
    
    best_gini = float('inf')
    best_split = [None, None] # [feature_index, threshold]

    for feature_idx in range(n_features):
        # 1. Obtener valores únicos ordenados de la característica actual
        unique_values = np.unique(X[:, feature_idx])
        
        # 2. Requisito: Probar umbrales en el punto medio entre valores consecutivos
        # Si solo hay un valor único, no se puede dividir por esta característica
        if len(unique_values) < 2:
            continue
            
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        
        for threshold in thresholds:
            # 3. Dividir los datos
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            # 4. Calcular Gini ponderado
            m_left = len(y_left)
            m_right = len(y_right)
            
            # Gini ponderado = (n_left/n)*Gini_left + (n_right/n)*Gini_right
            current_weighted_gini = (m_left / n_samples) * calculate_gini(y_left) + \
                                   (m_right / n_samples) * calculate_gini(y_right)
            
            # 5. Guardar la mejor división (menor impureza)
            # Nota: Usamos una pequeña tolerancia para estabilidad numérica
            if current_weighted_gini < best_gini - 1e-15:
                best_gini = current_weighted_gini
                best_split = [feature_idx, threshold]
                
    return best_split