import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # 1. Accuracy global
    accuracy = np.mean(y_true == y_pred)
    
    # 2. Métricas por clase
    metrics_per_class = {}
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        metrics_per_class[cls] = {"p": p, "r": r, "f1": f1, "sup": support}

    # 3. Agregación según el tipo de promedio
    if average == "binary":
        m = metrics_per_class.get(pos_label, {"p": 0.0, "r": 0.0, "f1": 0.0})
        precision, recall, f1 = m["p"], m["r"], m["f1"]
        
    elif average == "micro":
        # En micro, P = R = F1 = Accuracy en clasificación de etiqueta única
        precision = recall = f1 = accuracy
        
    elif average == "macro":
        # Promedio simple de las métricas de cada clase
        precision = np.mean([m["p"] for m in metrics_per_class.values()])
        recall = np.mean([m["r"] for m in metrics_per_class.values()])
        f1 = np.mean([m["f1"] for m in metrics_per_class.values()])
        
    elif average == "weighted":
        # Promedio ponderado por el soporte de cada clase
        total_sup = len(y_true)
        precision = sum(m["p"] * m["sup"] for m in metrics_per_class.values()) / total_sup
        recall = sum(m["r"] * m["sup"] for m in metrics_per_class.values()) / total_sup
        f1 = sum(m["f1"] * m["sup"] for m in metrics_per_class.values()) / total_sup

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

# --- Verificación con el caso del error ---
y_t = [0, 1, 2, 2]
y_p = [0, 1, 0, 2]
print(classification_metrics(y_t, y_p, average="macro"))
# Resultado esperado para f1: (1.0 + 1.0 + 0.333) / 3 = 0.777778