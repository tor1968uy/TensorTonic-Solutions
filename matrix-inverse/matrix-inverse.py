import numpy as np

def matrix_inverse(A):
    """
    Calcula la inversa de una matriz cuadrada A de tamaño (n, n).
    """
    # 1. Validación de entrada y conversión a array
    try:
        A = np.asarray(A, dtype=float)
    except (ValueError, TypeError):
        return None

    # 2. Verificar que sea 2D y cuadrada
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    
    # 3. Comprobar si la matriz es singular
    # Una matriz es singular si su determinante es 0. 
    # Usamos un umbral (epsilon) para manejar la precisión de punto flotante.
    try:
        det = np.linalg.det(A)
        if np.abs(det) < 1e-10:
            return None
            
        # 4. Calcular la inversa
        A_inv = np.linalg.inv(A)
        
        # 5. Verificación de precisión: ||A @ A_inv - I|| < 10^-7
        n = A.shape[0]
        identity = np.eye(n)
        error = np.linalg.norm(np.dot(A, A_inv) - identity)
        
        if error > 1e-7:
            # En casos de matrices mal condicionadas, el error podría ser mayor
            return None
            
        return A_inv

    except np.linalg.LinAlgError:
        # Captura errores específicos de álgebra lineal (como matrices no invertibles)
        return None

# --- Ejemplos de uso ---
A1 = [[1, 2], [3, 4]]
print(f"Inversa de 2x2:\n{matrix_inverse(A1)}")

A2 = [[2.0]]
print(f"\nInversa de 1x1:\n{matrix_inverse(A2)}")

A_singular = [[1, 2], [2, 4]]
print(f"\nMatriz singular (1,2 / 2,4): {matrix_inverse(A_singular)}")