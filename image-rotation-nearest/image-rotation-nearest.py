import numpy as np
import math

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    Returns a 2D list.
    """
    image_np = np.array(image)
    H, W = image_np.shape
    output = np.zeros_like(image_np)
    
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    
    theta = math.radians(angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    for i in range(H):
        for j in range(W):
            dy = i - cy
            dx = j - cx
            
            src_y = cy + (dy * cos_t) + (dx * sin_t)
            src_x = cx - (dy * sin_t) + (dx * cos_t)
            
            sy = int(round(src_y))
            sx = int(round(src_x))
            
            if 0 <= sy < H and 0 <= sx < W:
                output[i, j] = image_np[sy, sx]
            else:
                output[i, j] = 0
                
    # Convert from NumPy array back to a nested Python list
    return output.tolist()