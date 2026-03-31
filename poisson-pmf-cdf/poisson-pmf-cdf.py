import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF using log-stability.
    
    Args:
        lam (float): Average rate (lambda).
        k (int): Number of occurrences.
        
    Returns:
        tuple: (pmf, cdf)
    """
    # Función auxiliar para calcular el logaritmo del factorial: ln(k!)
    # ln(k!) = ln(1 * 2 * ... * k) = ln(1) + ln(2) + ... + ln(k)
    def log_factorial(n):
        if n <= 0:
            return 0.0
        return np.sum(np.log(np.arange(1, n + 1)))

    # 1. Calcular PMF de forma estable
    # log_pmf = k*ln(lam) - lam - ln(k!)
    log_pmf_k = k * np.log(lam) - lam - log_factorial(k)
    pmf = np.exp(log_pmf_k)

    # 2. Calcular CDF: Sumatoria de PMFs desde i=0 hasta k
    # Calculamos todos los log_factoriales de 0 a k de una vez para ser eficientes
    # Usamos np.cumsum sobre los logaritmos de los números para obtener ln(i!)
    if k == 0:
        cdf = pmf
    else:
        # Creamos un array de 0 a k
        indices = np.arange(k + 1)
        
        # Calculamos ln(i!) para cada i en el rango
        # log_fact_array[0] será 0, log_fact_array[1] será ln(1), etc.
        log_fact_array = np.zeros(k + 1)
        if k > 0:
            log_fact_array[1:] = np.cumsum(np.log(np.arange(1, k + 1)))
        
        # Calculamos el log_pmf para cada i de 0 a k
        log_pmfs = indices * np.log(lam) - lam - log_fact_array
        
        # Convertimos de vuelta y sumamos
        cdf = np.sum(np.exp(log_pmfs))

    return float(pmf), float(cdf)

# --- Verificación con ejemplos ---
examples = [(3, 2), (2.5, 0), (1.0, 1)]
for l, val_k in examples:
    p, c = poisson_pmf_cdf(l, val_k)
    print(f"Input: lam={l}, k={val_k} -> pmf≈{p:.4f}, cdf≈{c:.4f}")