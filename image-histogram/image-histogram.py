def image_histogram(image):
    """
    Compute the intensity histogram of a 2D grayscale image.
    Returns: a list of exactly 256 integers.
    """
    # 1. Initialize a list of 256 zeros (one for each intensity bin)
    histogram = [0] * 256
    
    # 2. Iterate through every row in the 2D image
    for row in image:
        # 3. Iterate through every pixel in the row
        for pixel in row:
            # The pixel value is the index into the histogram
            # pixel must be an integer between 0 and 255
            histogram[pixel] += 1
            
    return histogram