import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    Returns a 2D list of floats.
    """
    # Convert to numpy arrays for efficient slicing and math
    img = np.asarray(image, dtype=float)
    kern = np.asarray(kernel, dtype=float)
    
    H, W = img.shape
    kh, kw = kern.shape
    
    # 1. Apply Zero Padding
    # pad_width is ((top, bottom), (left, right))
    if padding > 0:
        img_padded = np.pad(img, pad_width=padding, mode='constant', constant_values=0)
    else:
        img_padded = img
        
    hp, wp = img_padded.shape
    
    # 2. Calculate Output Dimensions
    # H_out = floor((H + 2p - kh) / s) + 1
    h_out = (H + 2 * padding - kh) // stride + 1
    w_out = (W + 2 * padding - kw) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    # 3. Slide the kernel over the padded image
    for i in range(h_out):
        for j in range(w_out):
            # Calculate the window boundaries in the padded image
            start_i = i * stride
            end_i = start_i + kh
            start_j = j * stride
            end_j = start_j + kw
            
            # Extract the current image patch
            patch = img_padded[start_i:end_i, start_j:end_j]
            
            # Compute element-wise product and sum
            output[i, j] = np.sum(patch * kern)
            
    return output.tolist()