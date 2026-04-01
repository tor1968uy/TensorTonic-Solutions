import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calcula y ordena los autovalores de una matriz cuadrada.
    """
    # 1. Validación de entrada y conversión a array
    try:
        arr = np.asarray(matrix)
    except (ValueError, TypeError):
        return None

    # 2. Comprobar dimensiones: debe ser 2D y cuadrada (N x N)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    
    # Manejo de matrices vacías (0x0)
    if arr.size == 0:
        return np.array([])

    # 3. Calcular autovalores usando la función optimizada de NumPy
    # np.linalg.eigvals es más eficiente que np.linalg.eig si no se requieren autovectores
    eigenvalues = np.linalg.eigvals(arr)

    # 4. Ordenamiento consistente: Parte Real -> Parte Imaginaria
    # np.lexsort usa las claves de atrás hacia adelante (primero la última de la lista)
    # Queremos ordenar por real (primaria) e imaginaria (secundaria)
    idx = np.lexsort((eigenvalues.imag, eigenvalues.real))
    
    return eigenvalues[idx]

# --- Verificación con ejemplos ---
# Caso 1: Matriz Real Estándar
m1 = [[4, 1], [2, 3]]
print(f"Ejemplo 1 (4x1...): {calculate_eigenvalues(m1)}") 
# Resultado esperado: [2., 5.]

# Caso 2: Matriz con Autovalores Imaginarios
m2 = [[0, -1], [1, 0]]
print(f"Ejemplo 2 (Rotación): {calculate_eigenvalues(m2)}")
# Resultado esperado: [0.-1.j, 0.+1.j] (u orden inverso según lexsort)

# Caso 3: Matriz No Cuadrada
m3 = [[1, 2, 3], [4, 5, 6]]
print(f"Ejemplo 3 (No cuadrada): {calculate_eigenvalues(m3)}")
# Resultado esperado: None