import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normaliza una matriz 2D a lo largo de un eje usando L1, L2 o Max norm.
    """
    try:
        # 1. Convertir a array de NumPy y validar que sea 2D
        mat = np.array(matrix, dtype=float)
        if mat.ndim != 2:
            return None
            
        # 2. Calcular la norma según el tipo solicitado
        if norm_type == 'l2':
            # Norma L2: sqrt(sum(x^2))
            norm = np.sqrt(np.sum(np.square(mat), axis=axis, keepdims=True))
        elif norm_type == 'l1':
            # Norma L1: sum(abs(x))
            norm = np.sum(np.abs(mat), axis=axis, keepdims=True)
        elif norm_type == 'max':
            # Norma Max: max(abs(x))
            norm = np.max(np.abs(mat), axis=axis, keepdims=True)
        else:
            return None # Tipo de norma no soportado

        # 3. Manejo de vectores cero para evitar división por cero
        # Si la norma es 0, la dejamos en 1 para que la división mantenga el 0 original
        norm = np.where(norm == 0, 1, norm)

        # 4. Aplicar la normalización mediante broadcasting
        return mat / norm

    except (ValueError, TypeError):
        # Manejo simple de errores para entradas no numéricas o inválidas
        return None

# --- Pruebas con los ejemplos ---
print("L2 Row-wise:\n", matrix_normalization([[3, 4], [1, 0]], axis=1, norm_type='l2'))
print("\nL1 Column-wise:\n", matrix_normalization([[1, 2], [3, 4]], axis=0, norm_type='l1'))
print("\nMax Row-wise:\n", matrix_normalization([[2, 8], [4, 2]], axis=1, norm_type='max'))