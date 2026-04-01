import math

def ndcg(relevance_scores, k):
    """
    Calcula el NDCG@k usando la fórmula de ganancia exponencial.
    """
    # 1. Ajustar k al tamaño de la lista si es necesario
    k = min(k, len(relevance_scores))
    
    # Función auxiliar para calcular DCG
    def calculate_dcg(scores, top_k):
        dcg = 0.0
        for i in range(top_k):
            rel = scores[i]
            # Fórmula de ganancia exponencial: (2^rel - 1)
            gain = (2 ** rel) - 1
            # Descuento logarítmico (posiciones 1-indexed: i + 1)
            # El divisor es log2(pos + 1)
            discount = math.log2(i + 2)
            dcg += gain / discount
        return dcg

    # 2. Calcular DCG del ranking actual
    current_dcg = calculate_dcg(relevance_scores, k)
    
    # 3. Calcular IDCG (Ideal DCG)
    # El ranking ideal es simplemente los scores ordenados de mayor a menor
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = calculate_dcg(ideal_scores, k)
    
    # 4. Normalización
    if ideal_dcg == 0:
        return 0.0
        
    return current_dcg / ideal_dcg

# --- Validación con ejemplos ---
# Ejemplo 1: Orden ideal
rel1 = [3, 2, 1, 0]
print(f"NDCG@4 (Ideal): {ndcg(rel1, 4)}") # Esperado: 1.0

# Ejemplo 2: Orden inverso
rel2 = [0, 1, 2, 3]
print(f"NDCG@4 (Inverso): {ndcg(rel2, 4):.4f}") # Esperado: ~0.5479