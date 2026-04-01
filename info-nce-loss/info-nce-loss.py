import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Calcula la pérdida InfoNCE para aprendizaje contrastivo.
    Z1, Z2: Matrices de embeddings de tamaño (N, D).
    """
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    N = Z1.shape[0]

    # 1. Calcular matriz de similitud (Producto punto de todos contra todos)
    # S[i, j] representa la similitud entre la muestra i de Z1 y la j de Z2
    S = np.dot(Z1, Z2.T) / temperature

    # 2. Estabilidad numérica: restar el máximo por fila antes de aplicar exp
    # Esto evita desbordamientos (overflow) al calcular e^x
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max

    # 3. Calcular el Log-Sum-Exp para el denominador
    exp_S = np.exp(S_stable)
    sum_exp_S = np.sum(exp_S, axis=1)
    log_sum_exp = S_max.flatten() + np.log(sum_exp_S)

    # 4. Extraer las similitudes de los pares positivos (la diagonal)
    # Los pares positivos son (Z1[i], Z2[i])
    pos_sim = np.diag(S)

    # 5. Cross-entropy: -log( exp(pos) / sum(exp_all) ) = log_sum_exp - pos_sim
    losses = log_sum_exp - pos_sim

    # Retornar el promedio escalar de la pérdida en el lote
    return np.mean(losses)

# --- Verificación con ejemplos ---
z1 = np.array([[1, 0], [0, 1]])
z2 = np.array([[1, 0], [0, 1]])
print(f"Perfect Alignment (temp=0.1): {info_nce_loss(z1, z2, 0.1):.4f}") 

z2_bad = np.array([[0, 1], [1, 0]])
print(f"Misaligned (temp=0.1): {info_nce_loss(z1, z2_bad, 0.1):.4f}")

print(f"Moderate (temp=1.0): {info_nce_loss(z1, z2, 1.0):.4f}")