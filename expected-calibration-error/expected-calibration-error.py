import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Calcula el ECE comparando la precisión real con la confianza predicha en bins.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    
    # 1. Definir los límites de los bins (ancho constante)
    # np.linspace crea n_bins + 1 puntos, definiendo n_bins intervalos.
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    
    for i in range(n_bins):
        # 2. Determinar qué muestras caen en el bin actual [bin_low, bin_high)
        bin_low = bin_boundaries[i]
        bin_high = bin_boundaries[i + 1]
        
        # Caso especial para el último bin: incluir el valor 1.0 [low, 1.0]
        if i == n_bins - 1:
            in_bin = (y_pred >= bin_low) & (y_pred <= bin_high)
        else:
            in_bin = (y_pred >= bin_low) & (y_pred < bin_high)
            
        # 3. Calcular métricas si el bin no está vacío
        if np.any(in_bin):
            bin_true = y_true[in_bin]
            bin_pred = y_pred[in_bin]
            
            # Precisión media (Accuracy) del bin
            bin_acc = np.mean(bin_true)
            # Confianza media (Confidence) del bin
            bin_conf = np.mean(bin_pred)
            
            # Diferencia absoluta ponderada por la cantidad de muestras
            bin_weight = len(bin_true) / n_samples
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return ece

# --- Verificación con ejemplos ---
# Ejemplo 1: Sobreconfianza (0.9 vs 0.5)
y_t1 = [1, 0, 1, 0]
y_p1 = [0.9, 0.9, 0.9, 0.9]
print(f"Ejemplo 1 (Sobreconfianza): {expected_calibration_error(y_t1, y_p1, 5)}") # Esperado: 0.4

# Ejemplo 2: Distribución en 2 bins
y_t2 = [0, 0, 1, 1, 0, 1, 1, 1]
y_p2 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
print(f"Ejemplo 2 (2 Bins): {expected_calibration_error(y_t2, y_p2, 2)}") # Esperado: 0.125