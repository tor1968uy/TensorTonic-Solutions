def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    n = len(grad)
    m = len(s_list)
    
    # q inicial es el gradiente actual
    q = list(grad)
    alphas = [0.0] * m
    rhos = [0.0] * m
    
    # 1. Bucle hacia atrás (Backward loop)
    # Procesa desde el historial más reciente (m-1) al más antiguo (0)
    for i in range(m - 1, -1, -1):
        s_i = s_list[i]
        y_i = y_list[i]
        
        rho_i = 1.0 / _dot(y_i, s_i)
        rhos[i] = rho_i
        
        alpha_i = rho_i * _dot(s_i, q)
        alphas[i] = alpha_i
        
        # q = q - alpha_i * y_i
        for j in range(n):
            q[j] -= alpha_i * y_i[j]
            
    # 2. Escalado inicial del Hessiano (H0)
    # Se basa en el par más reciente (índice m-1)
    s_last = s_list[-1]
    y_last = y_list[-1]
    gamma = _dot(s_last, y_last) / _dot(y_last, y_last)
    
    # r inicial = gamma * q
    r = [gamma * qi for qi in q]
    
    # 3. Bucle hacia adelante (Forward loop)
    # Procesa desde el historial más antiguo (0) al más reciente (m-1)
    for i in range(m):
        s_i = s_list[i]
        y_i = y_list[i]
        
        beta = rhos[i] * _dot(y_i, r)
        
        # r = r + s_i * (alpha_i - beta)
        factor = alphas[i] - beta
        for j in range(n):
            r[j] += factor * s_i[j]
            
    # 4. Retornar la dirección de descenso (negativo de r)
    return [-ri for ri in r]

# --- Verificación con ejemplos ---
# Ejemplo 1D
g1 = [2.0]
s1 = [[1.0]]
y1 = [[2.0]]
print(f"Ejemplo 1D: {lbfgs_direction(g1, s1, y1)}") # Esperado: [-1.0]

# Ejemplo Identidad 3D
g2 = [6.0, 4.0, 2.0]
s2 = [[1,0,0],[0,1,0],[0,0,1]]
y2 = [[2,0,0],[0,2,0],[0,0,2]]
print(f"Ejemplo 3D: {lbfgs_direction(g2, s2, y2)}") # Esperado: [-3.0, -2.0, -1.0]