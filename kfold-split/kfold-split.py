import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Divide N elementos en k folds para validación cruzada.
    
    Args:
        N: Número total de muestras.
        k: Número de folds.
        shuffle: Si se deben barajar los índices antes de dividir.
        rng: Generador de números aleatorios de NumPy (opcional).
        
    Returns:
        list: Lista de k tuplas (train_indices, val_indices).
    """
    # 1. Crear el array de índices base
    indices = np.arange(N)
    
    # 2. Barajar si se solicita
    if shuffle:
        if rng is not None:
            # Uso de permutation para no modificar el original y ser determinista con rng
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)
            
    # 3. Dividir los índices en k carpetas (folds)
    # np.array_split maneja automáticamente cuando N no es divisible por k,
    # poniendo los elementos sobrantes en los primeros folds (N % k).
    folds = np.array_split(indices, k)
    
    results = []
    
    # 4. Generar las combinaciones de entrenamiento y validación
    for i in range(k):
        # El fold actual es para validación
        val_idx = folds[i]
        
        # El resto de los folds se concatenan para entrenamiento
        # Excluimos el índice i y unimos el resto
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        results.append((train_idx, val_idx))
        
    return results

# --- Validación con ejemplos ---
# Ejemplo 1: N=5, k=2, shuffle=False
print("Ejemplo N=5, k=2:")
for train, val in kfold_split(5, 2, shuffle=False):
    print(f"Train: {train}, Val: {val}")

# Ejemplo 2: N=7, k=3, shuffle=False
print("\nEjemplo N=7, k=3:")
for train, val in kfold_split(7, 3, shuffle=False):
    print(f"Train: {train}, Val: {val}")