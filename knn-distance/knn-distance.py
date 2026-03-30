import numpy as np

def knn_distance(X_train, X_test, k):
    # 1. Asegurar que sean arrays de NumPy
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # 2. Requisito Crítico: Convertir 1D a columnas (n muestras, 1 feature)
    # Si X_train es [1, 3, 5], debe ser [[1], [3], [5]]
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # 3. Calcular distancias con Broadcasting
    # (n_test, 1, d) - (1, n_train, d)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)

    # 4. Obtener índices ordenados
    # np.argsort es estable, mantendrá el orden de los índices en caso de empate
    sorted_indices = np.argsort(dist_sq, axis=1)

    # 5. Manejo de k y padding con -1
    # El sistema espera (n_test, k)
    result = np.full((n_test, k), -1, dtype=int)
    
    # Tomamos el mínimo entre k y n_train para no desbordar
    cols_to_fill = min(k, n_train)
    result[:, :cols_to_fill] = sorted_indices[:, :cols_to_fill]

    return result