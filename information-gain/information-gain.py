import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)
    n_total = y.size
    
    # 1. Calcular Entropía del nodo padre H(y)
    h_parent = _entropy(y)
    
    # 2. Dividir etiquetas usando la máscara
    y_left = y[split_mask]
    y_right = y[~split_mask]
    
    n_left = y_left.size
    n_right = y_right.size
    
    # Requisito: Si un lado está vacío, la ganancia es 0.0 (Hint 2)
    if n_left == 0 or n_right == 0:
        return 0.0
    
    # 3. Calcular Entropía de los hijos
    h_left = _entropy(y_left)
    h_right = _entropy(y_right)
    
    # 4. Calcular Entropía Condicional (Promedio ponderado)
    # H(y|split) = (n_l/n)*H(y_l) + (n_r/n)*H(y_r)
    h_children = (n_left / n_total) * h_left + (n_right / n_total) * h_right
    
    # Ganancia de Información = H(padre) - H(hijos)
    ig = h_parent - h_children
    
    return float(ig)