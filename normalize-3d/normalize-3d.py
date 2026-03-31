import numpy as np

def normalize_3d(v):
    """
    Normaliza vector(es) 3D a longitud unitaria de forma vectorizada.
    
    Args:
        v: Puede ser una lista [x, y, z], un array (3,) o un array (N, 3).
        
    Returns:
        np.ndarray: Vector(es) normalizados con el mismo shape que la entrada.
    """
    # Convertimos la entrada a un array de NumPy de tipo flotante
    v = np.array(v, dtype=float)
    
    # Guardamos el shape original para asegurar que el retorno sea consistente
    original_shape = v.shape
    
    # Si es un solo vector (3,), lo tratamos como (1, 3) internamente para generalizar
    if v.ndim == 1:
        v = v.reshape(1, -1)
    
    # 1. Calcular la norma de cada vector a lo largo del último eje (axis=1)
    # keepdims=True es vital para que la norma tenga forma (N, 1) y permita división directa
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    
    # 2. Crear una máscara para identificar vectores no nulos (evitar división por cero)
    # Usamos una tolerancia de 1e-10 como sugiere el hint
    nonzero_mask = (norms > 1e-10).reshape(-1)
    
    # 3. Inicializar el array de salida con ceros
    result = np.zeros_like(v)
    
    # 4. Dividir solo los vectores que tienen magnitud significativa
    # El broadcasting de NumPy se encarga de aplicar la norma (N, 1) a cada componente (N, 3)
    result[nonzero_mask] = v[nonzero_mask] / norms[nonzero_mask]
    
    # Retornamos al shape original (si entró (3,), sale (3,); si entró (N, 3), sale (N, 3))
    return result.reshape(original_shape)

# --- Pruebas de validación ---
print(f"Single vector [3, 4, 0]:\n{normalize_3d([3, 4, 0])}")
print(f"Batch [[0, 0, 0], [1, 2, 2]]:\n{normalize_3d([[0, 0, 0], [1, 2, 2]])}")