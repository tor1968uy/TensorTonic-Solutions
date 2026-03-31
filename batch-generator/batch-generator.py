import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Generador que baraja el dataset y devuelve mini-lotes (X_batch, y_batch).
    """
    # 1. Convertir a arrays de NumPy sin modificar los originales (copia superficial)
    X = np.array(X)
    y = np.array(y)
    n_samples = X.shape[0]
    
    # 2. Crear un array de índices y barajarlo
    indices = np.arange(n_samples)
    if rng is None:
        np.random.shuffle(indices)
    else:
        # Usamos el generador de números aleatorios proporcionado para reproducibilidad
        rng.shuffle(indices)
        
    # 3. Determinar el límite del bucle según drop_last
    # Si drop_last es True, ignoramos los elementos que no completan un lote
    if drop_last:
        upper_limit = n_samples - (n_samples % batch_size)
    else:
        upper_limit = n_samples

    # 4. Iterar sobre los índices en saltos de batch_size
    for i in range(0, upper_limit, batch_size):
        # Obtener los índices para el lote actual
        batch_indices = indices[i : i + batch_size]
        
        # Ceder (yield) el lote de X e y
        yield X[batch_indices], y[batch_indices]

# --- Ejemplos de Validación ---
X_test = np.arange(7)
y_test = np.arange(7)

print("--- drop_last = False ---")
for xb, yb in batch_generator(X_test, y_test, batch_size=3, drop_last=False):
    print(f"X_batch: {xb}, y_batch: {yb}")

print("\n--- drop_last = True ---")
for xb, yb in batch_generator(X_test, y_test, batch_size=3, drop_last=True):
    print(f"X_batch: {xb}, y_batch: {yb}")