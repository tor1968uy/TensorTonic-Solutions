import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Aseguramos que los puntos sean un array de NumPy
    points = np.asarray(points, dtype=float)
    
    # Hint 1: Manejar punto único (3,) convirtiéndolo a (1,3) para vectorizar
    is_single_point = points.ndim == 1
    if is_single_point:
        points = points.reshape(1, 3)
    
    # Hint 2: Convertir a coordenadas homogéneas (N, 4) añadiendo una columna de 1s
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Aplicar la transformación: (T @ points_h.T).T 
    # Esto multiplica la matriz de transformación por cada vector columna
    transformed_h = (T @ points_h.T).T
    
    # Dropear la última coordenada (el 1) para volver a 3D (N, 3)
    result = transformed_h[:, :3]
    
    # Si la entrada era un punto único, devolvemos la forma original (3,)
    if is_single_point:
        return result.ravel()
    
    return result