import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    Returns: List of [x1, y1, x2, y2]
    """
    anchors = []
    # Hint 1: El stride define cuánto espacio de la imagen cubre cada celda del feature map
    stride = image_size / feature_size
    
    # Recorremos la cuadrícula (row-major order: primero filas i, luego columnas j)
    for i in range(feature_size):
        for j in range(feature_size):
            # Centro de la celda con el offset de 0.5
            center_x = (j + 0.5) * stride
            center_y = (i + 0.5) * stride
            
            # Generar un anchor por cada combinación de escala y ratio
            for s in scales:
                for r in aspect_ratios:
                    # Hint 2: Cálculo de dimensiones manteniendo el área proporcional a s*s
                    # w = s * sqrt(r), h = s / sqrt(r)
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)
                    
                    # Convertir centro y dimensiones a coordenadas de esquina [x1, y1, x2, y2]
                    x1 = float(center_x - w / 2)
                    y1 = float(center_y - h / 2)
                    x2 = float(center_x + w / 2)
                    y2 = float(center_y + h / 2)
                    
                    anchors.append([x1, y1, x2, y2])
                    
    return anchors