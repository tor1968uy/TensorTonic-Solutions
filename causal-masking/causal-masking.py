import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    Aplica una máscara causal a una matriz de puntuaciones (scores).
    Mantiene la parte triangular inferior (incluyendo la diagonal) y 
    enmascara la parte superior.
    """
    # 1. Convertir a array de numpy (asegurando float para el mask_value)
    scores = np.array(scores, dtype=float)
    
    # 2. Obtener la dimensión temporal T (las dos últimas dimensiones son T x T)
    T = scores.shape[-1]
    
    # 3. Crear la máscara booleana para la parte triangular superior
    # np.triu con k=1 devuelve 1s por encima de la diagonal principal
    # Usamos dtype=bool para que ocupe menos memoria
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    
    # 4. Crear una copia para no modificar el input original (in-place)
    masked_scores = np.copy(scores)
    
    # 5. Aplicar la máscara usando broadcasting
    # NumPy aplicará automáticamente la máscara (T, T) a todas las dimensiones
    # precedentes (batch, heads, etc.) sin necesidad de bucles.
    masked_scores[..., mask] = mask_value
    
    return masked_scores

# --- Ejemplos de validación ---
scores_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("Input 2D:\n", np.array(scores_2d))
print("Output 2D:\n", apply_causal_mask(scores_2d))

# Ejemplo con dimensiones extra (Batch=1, Heads=2, T=3, T=3)
scores_4d = np.random.randn(1, 2, 3, 3)
output_4d = apply_causal_mask(scores_4d)
print("\nShape del output 4D:", output_4d.shape)