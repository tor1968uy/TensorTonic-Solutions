def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases 
    with full floating-point precision.
    """
    # 1. Obtener todas las calificaciones no nulas para la media global
    all_ratings = []
    for row in ratings_matrix:
        for val in row:
            if val != 0:
                all_ratings.append(val)
    
    if not all_ratings:
        return [0.0] * len(target_pairs)
        
    mu = sum(all_ratings) / len(all_ratings)
    
    num_users = len(ratings_matrix)
    num_items = len(ratings_matrix[0])
    
    # 2. Calcular sesgos de usuario (User Biases)
    user_biases = [0.0] * num_users
    for u in range(num_users):
        user_ratings = [val for val in ratings_matrix[u] if val != 0]
        if user_ratings:
            user_mean = sum(user_ratings) / len(user_ratings)
            user_biases[u] = user_mean - mu
            
    # 3. Calcular sesgos de ítem (Item Biases)
    item_biases = [0.0] * num_items
    for i in range(num_items):
        item_ratings = [ratings_matrix[u][i] for u in range(num_users) if ratings_matrix[u][i] != 0]
        if item_ratings:
            item_mean = sum(item_ratings) / len(item_ratings)
            item_biases[i] = item_mean - mu
            
    # 4. Generar predicciones finales
    predictions = []
    for u, i in target_pairs:
        # Nota: No redondear para mantener la precisión esperada por el validador
        pred = mu + user_biases[u] + item_biases[i]
        predictions.append(pred)
        
    return predictions

# --- Verificación con el caso de error ---
matrix = [[5, 3, 0], [4, 0, 1], [0, 1, 5]]
pairs = [[0, 2], [1, 1], [2, 0]]
# El resultado ahora incluirá los decimales largos (ej. 3.8333333333333335)
print(baseline_predict(matrix, pairs))