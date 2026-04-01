def linear_interpolation(values):
    n = len(values)
    result = list(values)
    
    # Encontrar el primer y último valor no-None
    i = 0
    while i < n:
        if result[i] is None:
            # Buscar el valor anterior conocido
            left_i = i - 1
            # Buscar el siguiente valor conocido
            right_i = i + 1
            while right_i < n and result[right_i] is None:
                right_i += 1
            
            # Interpolar
            if left_i < 0:
                # Sin valor anterior: rellenar con el primero conocido
                for k in range(i, right_i):
                    result[k] = result[right_i]
            elif right_i >= n:
                # Sin valor posterior: rellenar con el último conocido
                for k in range(i, n):
                    result[k] = result[left_i]
            else:
                # Interpolación lineal entre left y right
                left_val = result[left_i]
                right_val = result[right_i]
                gap = right_i - left_i
                for k in range(i, right_i):
                    t = (k - left_i) / gap
                    result[k] = left_val + t * (right_val - left_val)
            
            i = right_i
        else:
            i += 1
    
    return result