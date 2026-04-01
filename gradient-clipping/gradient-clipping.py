import numpy as np

def clip_gradients(g, max_norm):
    """
    Recorta los gradientes basándose en la norma L2 global.
    """
    # 1. Convertir a array de NumPy y asegurar que sea de tipo float
    g = np.asarray(g, dtype=float)
    
    # 2. Calcular la norma L2 global (norma de todos los elementos)
    # np.linalg.norm por defecto calcula la norma de Frobenius para matrices 
    # o la norma L2 para vectores, lo cual es equivalente a la raíz de la suma de cuadrados.
    total_norm = np.linalg.norm(g)
    
    # 3. Manejo de casos especiales
    # Si la norma es 0 (no hay gradiente) o max_norm no es válido, retornamos el original
    if total_norm == 0 or max_norm <= 0:
        return g.copy()
    
    # 4. Verificar si es necesario el recorte
    if total_norm > max_norm:
        # Calculamos el factor de escala: max_norm / total_norm
        scale_factor = max_norm / total_norm
        # Aplicamos el factor a todo el array para preservar la dirección
        return g * scale_factor
    
    # Si la norma es menor o igual, retornamos una copia sin cambios
    return g.copy()

# --- Verificación con ejemplos ---
# Ejemplo 1: No requiere recorte (Norma = 0.3)
g1 = [0.1, 0.2, 0.2]
print(f"Ejemplo 1 (Sin cambio): {clip_gradients(g1, 1.0)}")

# Ejemplo 2: Requiere recorte (Norma = 10.0, Max = 5.0 -> Escala 0.5)
g2 = [6, 8]
print(f"Ejemplo 2 (Escala 0.5): {clip_gradients(g2, 5.0)}")

# Ejemplo 3: Matriz 2D (Norma = 4.0, Max = 2.0 -> Escala 0.5)
g3 = [[2, 2], [2, 2]]
print(f"Ejemplo 3 (Matriz):\n{clip_gradients(g3, 2.0)}")