import math

def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    Returns a 2D list of floats representing gradient magnitude.
    """
    H = len(image)
    W = len(image[0])
    
    # 1. Define the Sobel kernels
    # Gx detects vertical edges (horizontal gradient)
    Gx_kernel = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    # Gy detects horizontal edges (vertical gradient)
    Gy_kernel = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]
    
    # 2. Create a zero-padded version of the image (1-pixel border)
    padded = [[0] * (W + 2) for _ in range(H + 2)]
    for i in range(H):
        for j in range(W):
            padded[i + 1][j + 1] = image[i][j]
            
    output = [[0.0] * W for _ in range(H)]
    
    # 3. Iterate through every pixel of the original image
    for i in range(H):
        for j in range(W):
            gx_val = 0.0
            gy_val = 0.0
            
            # Convolve with the 3x3 neighborhood in the padded image
            for ki in range(3):
                for kj in range(3):
                    # Pixel from the padded image
                    pixel = padded[i + ki][j + kj]
                    
                    gx_val += pixel * Gx_kernel[ki][kj]
                    gy_val += pixel * Gy_kernel[ki][kj]
            
            # 4. Compute gradient magnitude: sqrt(Gx^2 + Gy^2)
            magnitude = math.sqrt(gx_val**2 + gy_val**2)
            output[i][j] = magnitude
            
    return output