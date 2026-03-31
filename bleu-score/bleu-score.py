import math
from collections import Counter

def bleu_score(candidate, reference, max_n):
    """
    Calcula el BLEU score para una traducción candidata frente a una referencia.
    """
    c_len = len(candidate)
    r_len = len(reference)
    
    if c_len == 0:
        return 0.0
    
    precisions = []
    
    for n in range(1, max_n + 1):
        # 1. Extraer n-gramas del candidato
        c_ngrams = [tuple(candidate[i:i+n]) for i in range(c_len - n + 1)]
        if not c_ngrams:
            precisions.append(0.0)
            continue
            
        c_counts = Counter(c_ngrams)
        
        # 2. Extraer n-gramas de la referencia
        r_ngrams = [tuple(reference[i:i+n]) for i in range(r_len - n + 1)]
        r_counts = Counter(r_ngrams)
        
        # 3. Calcular precisión modificada (Clipped Count)
        # Sumamos cuántas veces aparece cada n-grama del candidato, 
        # pero limitado por su frecuencia en la referencia.
        clipped_hits = 0
        for ngram, count in c_counts.items():
            clipped_hits += min(count, r_counts.get(ngram, 0))
            
        precision_n = clipped_hits / len(c_ngrams)
        precisions.append(precision_n)
    
    # 4. Manejo de precisiones cero (BLEU se vuelve 0 si algún n-grama no coincide)
    if any(p == 0 for p in precisions):
        return 0.0
        
    # 5. Media geométrica de las precisiones
    # exp(sum(log(p_n)) / max_n)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    
    # 6. Calcular Brevity Penalty (BP)
    # Si c_len > r_len, BP = 1. Si c_len <= r_len, BP = exp(1 - r_len/c_len)
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1 - r_len / c_len)
        
    return bp * geo_mean

# --- Pruebas con los ejemplos ---
test1 = bleu_score(
    ["the", "cat", "sat", "on", "the", "mat"], 
    ["the", "cat", "sat", "on", "the", "mat"], 4
)
print(f"Test 1 (Perfecto): {test1:.1f}")

test2 = bleu_score(
    ["the", "cat", "is", "here"], 
    ["the", "cat", "was", "here"], 2
)
print(f"Test 2 (Parcial): {test2:.1f}")