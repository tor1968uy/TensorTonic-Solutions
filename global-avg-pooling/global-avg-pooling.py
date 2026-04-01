import numpy as np

def global_avg_pool(x):
    """
    Realiza Global Average Pooling sobre las dimensiones espaciales (H, W).
    Soporta tensores de rango 3 (C, H, W) y rango 4 (N, C, H, W).
    """
    # 1. Validación de entrada
    x = np.asanyarray(x)
    ndim = x.ndim
    
    if ndim not in [3, 4]:
        raise ValueError(f"Se esperaba un tensor de 3 o 4 dimensiones, pero se recibió uno de {ndim}.")

    # 2. Definir los ejes sobre los cuales promediar
    # Si es (C, H, W), los ejes espaciales son (1, 2)
    # Si es (N, C, H, W), los ejes espaciales son (2, 3)
    spatial_axes = (ndim - 2, ndim - 1)
    
    # 3. Calcular el promedio sobre los ejes espaciales
    # np.mean es vectorizado y eficiente. No modificamos el input original.
    # El resultado tendrá forma (C,) para rango 3 o (N, C) para rango 4.
    result = np.mean(x, axis=spatial_axes, dtype=np.float64)
    
    return result

# --- Validación con ejemplos ---
# Ejemplo 1: (C, H, W)
x1 = np.ones((3, 2, 2))
print(f"Input (3, 2, 2) -> Output: {global_avg_pool(x1)}") # Esperado: [1., 1., 1.]

# Ejemplo 2: (N, C, H, W)
x2 = np.array([[[[1, 2], [3, 4]]]]) # Shape (1, 1, 2, 2)
print(f"Input (1, 1, 2, 2) -> Output: {global_avg_pool(x2)}") # Esperado: [[2.5]]