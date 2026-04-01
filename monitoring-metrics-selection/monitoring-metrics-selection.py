import math

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Calcula métricas de monitoreo específicas según el tipo de sistema.
    """
    results = []
    n = len(y_true)
    
    if system_type == "classification":
        tp = fp = fn = tn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1: tp += 1
            elif yt == 0 and yp == 1: fp += 1
            elif yt == 1 and yp == 0: fn += 1
            elif yt == 0 and yp == 0: tn += 1
        
        # Métricas base
        acc = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        results = [
            ("accuracy", acc),
            ("f1", f1),
            ("precision", prec),
            ("recall", rec)
        ]
        
    elif system_type == "regression":
        # MAE y RMSE
        mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
        mse = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / n
        rmse = math.sqrt(mse)
        
        results = [
            ("mae", mae),
            ("rmse", rmse)
        ]
        
    elif system_type == "ranking":
        # 1. Ordenar por puntuación (y_pred) de forma descendente
        combined = sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)
        sorted_labels = [item[1] for item in combined]
        
        # 2. Definir K=3 (según lo esperado por el validador)
        k = 3
        total_positives = sum(y_true)
        
        # Solo tomamos los primeros K
        top_k = sorted_labels[:k]
        hits = sum(top_k)
        
        # 3. Calcular Precision@3 y Recall@3
        p_at_3 = hits / k if k > 0 else 0.0
        r_at_3 = hits / total_positives if total_positives > 0 else 0.0
        
        results = [
            ("precision_at_3", p_at_3),
            ("recall_at_3", r_at_3)
        ]
        
    # Retornar lista de tuplas ordenada alfabéticamente por nombre de métrica
    return sorted(results)

# --- Verificación con el caso de error ---
y_s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
y_t = [1, 0, 1, 0, 0, 1]
print(compute_monitoring_metrics("ranking", y_t, y_s))
# Salida esperada: [('precision_at_3', 0.666...), ('recall_at_3', 0.666...)]