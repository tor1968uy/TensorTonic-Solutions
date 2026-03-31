import numpy as np

def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Aplica calibración de regresión isotónica utilizando el algoritmo PAVA.
    """
    # 1. Preparación y ordenamiento de los datos de calibración
    cal_labels = np.array(cal_labels, dtype=float)
    cal_probs = np.array(cal_probs, dtype=float)
    new_probs = np.array(new_probs, dtype=float)
    
    # Ordenar por probabilidad predicha (eje X)
    idx = np.argsort(cal_probs)
    sorted_probs = cal_probs[idx]
    sorted_labels = cal_labels[idx]
    
    n = len(sorted_labels)
    
    # 2. Algoritmo PAVA (Pool Adjacent Violators Algorithm)
    # Estructura de bloques: [valor_promedio, peso (n_elementos), índice_inicio, índice_fin]
    blocks = []
    for i in range(n):
        # Crear un nuevo bloque para el elemento i
        new_block = {
            'val': sorted_labels[i],
            'size': 1
        }
        
        # Mientras el nuevo bloque sea menor que el anterior, fusionar (viola monotonicidad)
        while blocks and blocks[-1]['val'] > new_block['val']:
            prev_block = blocks.pop()
            # Calcular promedio ponderado
            combined_size = prev_block['size'] + new_block['size']
            combined_val = (prev_block['val'] * prev_block['size'] + 
                            new_block['val'] * new_block['size']) / combined_size
            new_block = {
                'val': combined_val,
                'size': combined_size
            }
        blocks.append(new_block)
    
    # 3. Expandir los bloques de nuevo a la longitud original n
    calibrated_y = []
    for block in blocks:
        calibrated_y.extend([block['val']] * block['size'])
    calibrated_y = np.array(calibrated_y)
    
    # 4. Interpolación lineal para new_probs
    # np.interp maneja automáticamente el "clamping" (valores fuera de rango toman los extremos)
    return np.interp(new_probs, sorted_probs, calibrated_y).tolist()

# --- Validación con el ejemplo de la consigna ---
cal_l = [0, 1, 0, 1]
cal_p = [0.1, 0.4, 0.6, 0.9]
new_p = [0.2, 0.5, 0.8]

result = calibrate_isotonic(cal_l, cal_p, new_p)
print(f"Calibrated probabilities: {[round(r, 4) for r in result]}")