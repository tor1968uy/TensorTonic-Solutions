import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-posterior probabilities for Bernoulli Naive Bayes.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    n_samples, n_features = X_train.shape
    classes = np.sort(np.unique(y_train))
    n_classes = len(classes)
    n_test = X_test.shape[0]
    
    # Inicializar matriz de salida (n_test, n_classes)
    log_posteriors = np.zeros((n_test, n_classes))
    alpha = 1  # Laplace smoothing
    
    for idx, c in enumerate(classes):
        # Filtrar muestras pertenecientes a la clase c
        X_c = X_train[y_train == c]
        class_count = X_c.shape[0]
        
        # 1. Log Prior: log(P(y)) = log(n_c / n)
        log_prior = np.log(class_count / n_samples)
        
        # 2. Probabilidades de características con suavizado de Laplace (Hint 2)
        # P(xi=1 | y) = (count(xi=1) + alpha) / (n_c + 2 * alpha)
        # Sumamos las ocurrencias de 1s en cada columna para la clase c
        p_i_1 = (np.sum(X_c, axis=0) + alpha) / (class_count + 2 * alpha)
        
        # Pre-calculamos los logs para eficiencia
        log_p_i_1 = np.log(p_i_1)
        log_p_i_0 = np.log(1 - p_i_1)
        
        # 3. Log Likelihood para cada muestra de test
        # Para cada feature: xi * log(P(xi=1|y)) + (1-xi) * log(P(xi=0|y))
        # Esto se puede vectorizar: X_test @ log_p_i_1 + (1 - X_test) @ log_p_i_0
        log_likelihood = X_test @ log_p_i_1 + (1 - X_test) @ log_p_i_0
        
        # Log Posterior = log(Prior) + log(Likelihood)
        log_posteriors[:, idx] = log_prior + log_likelihood
        
    return log_posteriors