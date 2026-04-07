import math

def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    Returns a 2D list of floats.
    """
    H = len(image)
    W = len(image[0])
    
    # Initialize the output grid
    output = [[0.0 for _ in range(new_w)] for _ in range(new_h)]
    
    for i in range(new_h):
        for j in range(new_w):
            # 1. Map output coordinates (i, j) to source coordinates (src_y, src_x)
            # Use the "align corners" formula
            if new_h > 1:
                src_y = i * (H - 1) / (new_h - 1)
            else:
                src_y = 0.0
                
            if new_w > 1:
                src_x = j * (W - 1) / (new_w - 1)
            else:
                src_x = 0.0
            
            # 2. Find the four neighboring integer coordinates
            y0 = int(math.floor(src_y))
            x0 = int(math.floor(src_x))
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)
            
            # 3. Calculate fractional offsets (weights)
            dy = src_y - y0
            dx = src_x - x0
            
            # 4. Interpolate across the four neighbors
            # Top-left weight: (1-dy)(1-dx)
            # Bottom-left weight: (dy)(1-dx)
            # Top-right weight: (1-dy)(dx)
            # Bottom-right weight: (dy)(dx)
            
            val = (image[y0][x0] * (1 - dy) * (1 - dx) +
                   image[y1][x0] * dy * (1 - dx) +
                   image[y0][x1] * (1 - dy) * dx +
                   image[y1][x1] * dy * dx)
            
            output[i][j] = float(val)
            
    return output