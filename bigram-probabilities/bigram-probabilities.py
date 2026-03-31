from collections import Counter

def bigram_probabilities(tokens):
    """
    Calcula conteos de bigramas y sus probabilidades con suavizado add-1.
    """
    # 1. Construir el vocabulario único y determinar su tamaño |V|
    vocabulary = sorted(list(set(tokens)))
    v_size = len(vocabulary)
    
    # 2. Contar todas las ocurrencias de bigramas (w1, w2)
    # Solo contamos las transiciones existentes en el texto original
    bigram_list = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
    counts = Counter(bigram_list)
    
    # 3. Contar la frecuencia de cada palabra como contexto (w1)
    # Esto servirá para el denominador: Count(w1)
    context_counts = Counter(tokens[:-1])
    
    # 4. Calcular probabilidades para TODO el espacio V x V (Requisito)
    probs = {}
    
    # Iteramos sobre cada palabra posible como contexto (w1)
    for w1 in vocabulary:
        # El denominador para w1 es: total de veces que aparece w1 + |V|
        # Nota: context_counts[w1] será 0 si la palabra solo aparece al final del texto
        denominator = context_counts[w1] + v_size
        
        # Para cada w1, calculamos la probabilidad de que le siga cualquier w2 en V
        for w2 in vocabulary:
            # Numerador: veces que vimos la secuencia (w1, w2) + 1 (smoothing)
            numerator = counts.get((w1, w2), 0) + 1
            probs[(w1, w2)] = numerator / denominator
            
    return dict(counts), probs

# --- Ejemplo de Validación ---
tokens_ex = ["a", "b", "a"]
cnts, prbs = bigram_probabilities(tokens_ex)

print(f"Vocabulario: {sorted(list(set(tokens_ex)))}")
print(f"Conteo ('a', 'b'): {cnts.get(('a', 'b'))}")
print(f"Probabilidad P('b' | 'a'): {prbs[('a', 'b')]:.3f}") # (1+1) / (1+2) = 0.667
print(f"Probabilidad P('a' | 'a'): {prbs[('a', 'a')]:.3f}") # (0+1) / (1+2) = 0.333