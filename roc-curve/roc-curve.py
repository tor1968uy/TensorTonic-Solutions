import numpy as np

def roc_curve(y_true, y_score):
    """
    Calcula la curva ROC de forma vectorizada.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 1. Ordenar por puntuación descendente
    # Usamos np.argsort con [::-1] para descendente
    desc_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_indices]
    y_true = y_true[desc_indices]

    # 2. Identificar dónde cambian las puntuaciones (para manejar empates)
    # Buscamos índices donde y_score[i] != y_score[i+1]
    distinct_indices = np.where(np.diff(y_score))[0]
    # El umbral final siempre se incluye
    threshold_indices = np.r_[distinct_indices, y_true.size - 1]

    # 3. Calcular acumulados de Positivos (TP) y Negativos (FP)
    # TP acumulados: suma de y_true (donde 1 es positivo)
    tps = np.cumsum(y_true)[threshold_indices]
    # FP acumulados: total de elementos vistos menos los positivos vistos
    fps = 1 + threshold_indices - tps

    # 4. Calcular tasas (TPR y FPR)
    # Evitar división por cero si no hay positivos o negativos
    total_pos = tps[-1]
    total_neg = fps[-1]
    
    tpr = tps / total_pos if total_pos > 0 else np.zeros_like(tps)
    fpr = fps / total_neg if total_neg > 0 else np.zeros_like(fps)

    # 5. Agregar el punto de inicio (0,0) con umbral inf
    # Requisito: fpr=[0, ...], tpr=[0, ...], thresholds=[inf, ...]
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[np.inf, y_score[threshold_indices]]

    return fpr, tpr, thresholds

# --- Validación con el ejemplo ---
if __name__ == "__main__":
    y_t = [0, 1]
    y_s = [0.1, 0.9]
    f, t, thr = roc_curve(y_t, y_s)
    print(f"FPR: {f}, TPR: {t}, Thresholds: {thr}")

    y_t2 = [1, 0, 1, 0]
    y_s2 = [0.9, 0.7, 0.4, 0.2]
    f2, t2, thr2 = roc_curve(y_t2, y_s2)
    print(f"\nEjemplo 2:\nFPR: {f2}\nTPR: {t2}\nThresholds: {thr2}")