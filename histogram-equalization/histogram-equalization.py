def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    Returns a 2D list of integers.
    """
    H = len(image)
    W = len(image[0])
    total_pixels = H * W
    
    # 1. Compute the histogram (frequencies of 0-255)
    hist = [0] * 256
    for row in image:
        for pixel in row:
            hist[pixel] += 1
            
    # 2. Compute the Cumulative Distribution Function (CDF)
    cdf = [0] * 256
    current_sum = 0
    for i in range(256):
        current_sum += hist[i]
        cdf[i] = current_sum
        
    # 3. Find cdf_min (the smallest non-zero value in the CDF)
    cdf_min = 0
    for value in cdf:
        if value > 0:
            cdf_min = value
            break
            
    # 4. Handle edge case: all pixels are the same
    if total_pixels == cdf_min:
        return [[0 for _ in range(W)] for _ in range(H)]
    
    # 5. Map each pixel to a new value
    # Formula: round((cdf[v] - cdf_min) / (total_pixels - cdf_min) * 255)
    output = []
    for row in image:
        new_row = []
        for v in row:
            # Applying the normalization formula
            numerator = cdf[v] - cdf_min
            denominator = total_pixels - cdf_min
            new_val = round((numerator / denominator) * 255)
            new_row.append(int(new_val))
        output.append(new_row)
            
    return output