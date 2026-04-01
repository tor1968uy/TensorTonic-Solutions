import numpy as np

def silhouette_score(X, labels):
    """
    Calcula el Silhouette Score medio para un conjunto de puntos y etiquetas.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return 0.0

    # 1. Calcular matriz de distancias Euclidianas de todos contra todos (N x N)
    # Usamos la identidad: ||u-v||^2 = ||u||^2 + ||v||^2 - 2(u·v)
    sq_norms = np.sum(X**2, axis=1).reshape(-1, 1)
    dist_matrix = np.sqrt(np.maximum(sq_norms + sq_norms.T - 2 * np.dot(X, X.T), 0))

    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)

    # 2. Calcular a(i) y b(i) por cada punto
    for label in unique_labels:
        # Máscara para puntos dentro del clúster actual y fuera de él
        in_cluster = (labels == label)
        out_cluster = ~in_cluster
        
        n_in = np.sum(in_cluster)
        
        # Distancia intra-clúster a(i)
        # Sumamos las distancias de cada punto i en 'label' a otros puntos en 'label'
        if n_in > 1:
            intra_dists = dist_matrix[in_cluster][:, in_cluster]
            a[in_cluster] = np.sum(intra_dists, axis=1) / (n_in - 1)
        else:
            a[in_cluster] = 0.0 # Un clúster de un solo punto tiene a(i) = 0

        # Distancia inter-clúster b(i)
        for other_label in unique_labels:
            if label == other_label:
                continue
            
            other_in_cluster = (labels == other_label)
            n_other = np.sum(other_in_cluster)
            
            # Promedio de distancias desde puntos de 'label' a puntos de 'other_label'
            inter_dists = dist_matrix[in_cluster][:, other_in_cluster]
            avg_inter = np.sum(inter_dists, axis=1) / n_other
            
            # b(i) es el mínimo de estos promedios hacia otros clústeres
            b[in_cluster] = np.minimum(b[in_cluster], avg_inter)

    # 3. Calcular el coeficiente de silueta para cada muestra
    silhouettes = (b - a) / np.maximum(a, b)
    
    # Manejar casos donde max(a,b) es 0 (puntos coincidentes o clúster único)
    silhouettes = np.nan_to_num(silhouettes)

    return float(np.mean(silhouettes))

# --- Validación ---
X = [[0,0],[0,1],[1,0],[5,5],[5,6],[6,5]]
labels = [0,0,0,1,1,1]
print(f"Silhouette Score: {silhouette_score(X, labels):.4f}") # Esperado ≈ 0.79