import numpy as np

def angle_between_3d(v, w):
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    
    cos_angle = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # evitar errores numéricos en arccos
    
    return float(np.arccos(cos_angle))