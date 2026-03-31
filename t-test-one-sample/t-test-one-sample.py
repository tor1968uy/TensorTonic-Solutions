import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Convertimos a array de numpy para asegurar operaciones vectorizadas
    x = np.array(x, dtype=float)
    
    # n es el tamaño de la muestra
    n = len(x)
    
    # Media muestral
    sample_mean = np.mean(x)
    
    # Desviación estándar con corrección de Bessel (ddof=1 -> n-1)
    # Si todos los valores son iguales, sample_std será 0.0
    sample_std = np.std(x, ddof=1)
    
    # Error estándar: s / sqrt(n)
    standard_error = sample_std / np.sqrt(n)
    
    # Calculamos el estadístico t
    # Si standard_error es 0 y (sample_mean - mu0) != 0, 
    # NumPy devolverá np.inf automáticamente por la división.
    t_stat = (sample_mean - mu0) / standard_error
    
    return float(t_stat)

# Prueba del caso que fallaba:
# x=[5, 5, 5, 5], mu0=4 -> t_stat: inf