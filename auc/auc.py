import numpy as np

def auc(fpr, tpr):
    """
    Calcula el AUC (Area Under Curve) utilizando la regla del trapecio.
    """
    # 1. Validación de entrada
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    
    if len(fpr) != len(tpr) or len(fpr) < 2:
        return None

    # 2. Integración mediante la regla del trapecio
    # La fórmula es sum((x_i+1 - x_i) * (y_i + y_i+1) / 2)
    # NumPy proporciona np.trapezoid (o np.trapz en versiones antiguas)
    # Nota: fpr es el eje x, tpr es el eje y.
    
    # Usamos np.trapezoid si está disponible (NumPy 2.0+), 
    # de lo contrario usamos np.trapz para compatibilidad.
    if hasattr(np, 'trapezoid'):
        area = np.trapezoid(y=tpr, x=fpr)
    else:
        area = np.trapz(y=tpr, x=fpr)
        
    # El AUC siempre se devuelve como un valor absoluto (el área no es negativa)
    # y típicamente el FPR va de 0 a 1, por lo que el área es positiva.
    return float(np.abs(area))

# --- Verificación con ejemplos ---
# Caso 1: Clasificador Perfecto (Escalón)
print(f"Perfecto: {auc([0, 0, 1], [0, 1, 1])}") # Esperado: 1.0

# Caso 2: Clasificador Aleatorio (Diagonal)
print(f"Aleatorio: {auc([0, 1], [0, 1])}")     # Esperado: 0.5

# Caso 3: Peor Clasificador (Inverso)
print(f"Peor: {auc([0, 1, 1], [0, 0, 1])}")    # Esperado: 0.0