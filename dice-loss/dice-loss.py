import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Convert inputs to numpy arrays and flatten to handle any shape (1D or 2D)
    p = np.asarray(p).flatten()
    y = np.asarray(y).flatten()
    
    # Calculate intersection: sum of element-wise product
    intersection = np.sum(p * y)
    
    # Calculate sums of p and y
    sum_p = np.sum(p)
    sum_y = np.sum(y)
    
    # Calculate Dice Coefficient with smoothing epsilon
    dice_coeff = (2. * intersection + eps) / (sum_p + sum_y + eps)
    
    # Dice Loss is 1 minus the coefficient
    return 1. - dice_coeff