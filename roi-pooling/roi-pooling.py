import numpy as np

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    feature_map: 2D list or array of shape (H, W)
    rois: list of [x1, y1, x2, y2]
    output_size: integer (target height and width are the same)
    """
    # Convertimos a array de numpy para facilitar el slicing
    feature_map = np.asarray(feature_map)
    all_pooled_outputs = []
    
    for x1, y1, x2, y2 in rois:
        roi_h = y2 - y1
        roi_w = x2 - x1
        
        # Inicializamos la rejilla de salida para esta ROI
        pooled_roi = np.zeros((output_size, output_size))
        
        for i in range(output_size):
            for j in range(output_size):
                # Aplicamos las fórmulas de los requerimientos:
                # hstart = y1 + floor(i * roi_h / output_size)
                h_start = y1 + (i * roi_h // output_size)
                h_end = y1 + ((i + 1) * roi_h // output_size)
                
                # wstart = x1 + floor(j * roi_w / output_size)
                w_start = x1 + (j * roi_w // output_size)
                w_end = x1 + ((j + 1) * roi_w // output_size)
                
                # Requisito: Asegurar que cada bin cubra al menos un píxel
                h_end = max(h_end, h_start + 1)
                w_end = max(w_end, w_start + 1)
                
                # Extraemos la sub-región (bin) y aplicamos Max Pooling
                bin_region = feature_map[h_start:h_end, w_start:w_end]
                
                # Si por alguna razón la región está vacía (fuera de límites), ponemos 0
                if bin_region.size > 0:
                    pooled_roi[i, j] = np.max(bin_region)
                else:
                    pooled_roi[i, j] = 0
                    
        # Convertimos el array de la ROI a lista para el formato de salida esperado
        all_pooled_outputs.append(pooled_roi.tolist())
        
    return all_pooled_outputs