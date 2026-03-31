import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Calcula la matriz de confusión con normalización opcional usando bincount.
    """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    
    # 1. Determinar el número de clases (K)
    if num_classes is None:
        if y_true.size == 0:
            return np.array([[]])
        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
    
    K = num_classes
    
    # 2. Calcular índices lineales: true * K + pred
    # Esto coloca cada par (true, pred) en una posición única entre 0 y K^2 - 1
    indices = y_true * K + y_pred
    
    # 3. Contar ocurrencias con bincount
    # minlength asegura que el array tenga tamaño K*K incluso si faltan clases
    counts = np.bincount(indices, minlength=K**2)
    cm = counts.reshape(K, K)
    
    # 4. Normalización
    if normalize == 'none':
        return cm.astype(np.int64)
    
    cm = cm.astype(np.float64)
    
    if normalize == 'true':
        # Normalizar por filas (verdaderos positivos reales por clase)
        # Sumamos horizontalmente: axis=1
        row_sums = np.sum(cm, axis=1, keepdims=True)
        # Reemplazar ceros por 1 para evitar division by zero (o usar epsilon)
        cm /= np.where(row_sums == 0, 1, row_sums)
        
    elif normalize == 'pred':
        # Normalizar por columnas (predicciones totales por clase)
        # Sumamos verticalmente: axis=0
        col_sums = np.sum(cm, axis=0, keepdims=True)
        cm /= np.where(col_sums == 0, 1, col_sums)
        
    elif normalize == 'all':
        # Normalizar por el total de muestras
        total_sum = np.sum(cm)
        if total_sum > 0:
            cm /= total_sum

    return cm

# --- Ejemplos de Validación ---
y_t = [0, 1, 1]
y_p = [0, 1, 0]

print("Normalización 'none':\n", confusion_matrix_norm(y_t, y_p, normalize='none'))
print("\nNormalización 'true':\n", confusion_matrix_norm(y_t, y_p, normalize='true'))