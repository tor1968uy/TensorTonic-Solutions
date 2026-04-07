import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    y_true: np.ndarray of shape (N,) containing class indices
    y_pred: np.ndarray of shape (N, K) containing predicted probabilities
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Get the number of samples
    N = y_true.shape[0]
    
    # Advanced Indexing: 
    # np.arange(N) creates [0, 1, ..., N-1]
    # y_true provides the specific column index for each row
    # This extracts p_i,y_i for all i samples
    correct_class_probs = y_pred[np.arange(N), y_true]
    
    # Compute the negative log for each selected probability
    losses = -np.log(correct_class_probs)
    
    # Return the average loss across all samples
    return float(np.mean(losses))