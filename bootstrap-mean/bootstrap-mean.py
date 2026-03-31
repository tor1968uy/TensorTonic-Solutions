import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Realiza un remuestreo Bootstrap para estimar la media y su intervalo de confianza.
    
    Args:
        x: Lista o array con los datos originales.
        n_bootstrap: Número de remuestreos (por defecto 1000).
        ci: Nivel de confianza (ej. 0.95 para 95%).
        rng: Generador de números aleatorios de NumPy (opcional).
        
    Returns:
        tuple: (boot_means, lower, upper)
    """
    # 1. Preparación de datos y generador
    x = np.array(x)
    n = len(x)
    if rng is None:
        rng = np.random.default_rng()

    # 2. Generación de remuestreos (Vectorizado)
    # Creamos una matriz de n_bootstrap filas y n columnas
    # Cada fila es una muestra aleatoria con reemplazo de x
    resamples = rng.choice(x, size=(n_bootstrap, n), replace=True)

    # 3. Calcular la estadística de interés (la media) para cada remuestreo
    boot_means = np.mean(resamples, axis=1)

    # 4. Calcular los límites del intervalo de confianza usando percentiles
    # Para un CI del 95%, buscamos los percentiles 2.5 y 97.5
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)

    return boot_means, float(lower), float(upper)

# Ejemplo de uso:
# data = [10, 12, 11, 15, 9, 13, 8]
# means, low, high = bootstrap_mean(data)
# print(f"Media original: {np.mean(data):.2f}")
# print(f"Intervalo de confianza 95%: [{low:.2f}, {high:.2f}]")