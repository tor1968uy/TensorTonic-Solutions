import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Calcula el F1 micro-promediado para etiquetas enteras multiclase.
    """
    # 1. Convertir a arrays de NumPy para operaciones eficientes
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 2. Calcular Verdaderos Positivos (TP) globales
    # En clasificación multiclase de etiqueta única, TP es simplemente
    # el número de veces que la predicción coincide con la realidad.
    tp = np.sum(y_true == y_pred)
    
    # 3. Calcular Falsos Positivos (FP) y Falsos Negativos (FN) globales
    # FP + FN es el total de errores en el sistema.
    # Como cada error de predicción genera un FP y un FN a nivel global:
    total_samples = len(y_true)
    errors = total_samples - tp
    
    fp = errors
    fn = errors
    
    # 4. Aplicar la fórmula de Micro-F1
    # F1 = 2*TP / (2*TP + FP + FN)
    denominator = (2 * tp) + fp + fn
    
    if denominator == 0:
        return 0.0
    
    f1 = (2 * tp) / denominator
    
    return float(f1)