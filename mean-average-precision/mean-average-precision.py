import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Computa el mAP para múltiples consultas de recuperación.
    
    Args:
        y_true_list: Lista de listas con etiquetas binarias (1=relevante).
        y_score_list: Lista de listas con puntuaciones de confianza.
        k: Entero opcional para limitar la evaluación (mAP@k).
        
    Returns:
        tuple: (mean_ap, list_of_ap_per_query)
    """
    ap_per_query = []
    
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # 1. Manejo de caso borde: no hay elementos relevantes
        if np.sum(y_true) == 0:
            ap_per_query.append(0.0)
            continue
            
        # 2. Ordenar etiquetas por puntuación descendente
        desc_indices = np.argsort(y_score)[::-1]
        if k is not None:
            desc_indices = desc_indices[:k]
            
        sorted_labels = y_true[desc_indices]
        
        # 3. Calcular Precisión en cada posición i
        # cumulative_rel: cuántos aciertos llevamos hasta la posición i
        cumulative_rel = np.cumsum(sorted_labels)
        # ranks: [1, 2, 3, ...]
        ranks = np.arange(1, len(sorted_labels) + 1)
        # precision_at_i: aciertos / total_vistos
        precisions = cumulative_rel / ranks
        
        # 4. Calcular el AP de la consulta
        # Solo promediamos las precisiones donde el elemento era realmente relevante
        relevant_precisions = precisions[sorted_labels == 1]
        
        if len(relevant_precisions) == 0:
            ap_per_query.append(0.0)
        else:
            # El divisor es el número total de elementos relevantes en el set original
            # (o el número de relevantes dentro de k si se prefiere esa variante, 
            # pero el estándar suele ser sobre el total de relevantes en y_true)
            actual_rel_count = np.sum(y_true)
            ap = np.sum(relevant_precisions) / actual_rel_count
            ap_per_query.append(ap)
            
    # 5. Calcular la media de todos los APs
    mean_ap = np.mean(ap_per_query) if ap_per_query else 0.0
    
    return float(mean_ap), ap_per_query

# --- Validación con ejemplos ---
y_t = [[1, 0, 1, 0]]
y_s = [[0.9, 0.8, 0.7, 0.1]]
mAP, aps = mean_average_precision(y_t, y_s)
print(f"Ejemplo 1: mAP = {mAP:.4f}, APs = {aps}")

y_t2 = [[1, 0, 1], [1, 1, 0]]
y_s2 = [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]]
mAP2, aps2 = mean_average_precision(y_t2, y_s2)
print(f"Ejemplo 2: mAP = {mAP2:.4f}, APs = {aps2}")