import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels using Gaussian Naive Bayes.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    classes = np.unique(y_train)
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    epsilon = 1e-9
    
    # Diccionarios para almacenar parámetros por clase
    class_stats = {}
    
    for c in classes:
        # Filtrar datos por clase
        X_c = X_train[y_train == c]
        
        # Calcular Prior P(c) = n_c / n
        prior = len(X_c) / n_samples
        
        # Calcular media y varianza poblacional (divide por n_c) por cada feature
        means = np.mean(X_c, axis=0)
        # Requisito: varianza poblacional + epsilon
        vars = np.var(X_c, axis=0) + epsilon
        
        class_stats[c] = {
            'prior': prior,
            'means': means,
            'vars': vars
        }
        
    predictions = []
    
    for x in X_test:
        posteriors = {}
        
        for c in classes:
            stats = class_stats[c]
            # log(P(c))
            log_prior = np.log(stats['prior'])
            
            # Hint 2: Suma de log-likelihoods gaussianas por feature
            # formula: -0.5 * log(2 * pi * var) - (x_j - mean)^2 / (2 * var)
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * stats['vars']) + 
                ((x - stats['means'])**2 / stats['vars'])
            )
            
            posteriors[c] = log_prior + log_likelihood
            
        # Predecir la clase con el log posterior más alto
        predictions.append(max(posteriors, key=posteriors.get))
        
    return [int(p) for p in predictions]