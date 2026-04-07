import numpy as np

def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    Returns a 2D list of integers (0 or 1).
    """
    img = np.asarray(image, dtype=int)
    kern = np.asarray(kernel, dtype=int)
    
    H, W = img.shape
    kh, kw = kern.shape
    
    # Calculate padding based on kernel size (assuming odd dimensions)
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Apply zero-padding to the input image
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    output = np.zeros((H, W), dtype=int)
    
    # Find the local indices where the kernel itself is 1
    # We only care about the image pixels under the 'active' parts of the kernel
    kernel_indices = np.argwhere(kern == 1)
    
    for i in range(H):
        for j in range(W):
            # Extract the window from the padded image corresponding to the kernel
            # The window starts at (i, j) in the padded coordinate system
            window = img_padded[i : i + kh, j : j + kw]
            
            # Get the image values only where the kernel is 1
            relevant_pixels = window[kern == 1]
            
            if operation == "erode":
                # Erosion: Output 1 only if ALL relevant pixels are 1
                if np.all(relevant_pixels == 1):
                    output[i, j] = 1
                else:
                    output[i, j] = 0
            
            elif operation == "dilate":
                # Dilation: Output 1 if ANY relevant pixel is 1
                if np.any(relevant_pixels == 1):
                    output[i, j] = 1
                else:
                    output[i, j] = 0
                    
    return output.tolist()