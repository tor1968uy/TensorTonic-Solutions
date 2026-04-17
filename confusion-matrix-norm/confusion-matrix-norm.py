import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle num_classes inference
    if num_classes is None:
        if y_true.size == 0:
            return np.array([[]], dtype=np.int64 if normalize == 'none' else np.float64)
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    # Initialize the matrix with the requested output type
    if normalize == 'none':
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    else:
        cm = np.zeros((num_classes, num_classes), dtype=np.float64)

    # Only perform bincount if there is data to process
    if y_true.size > 0:
        indices = y_true * num_classes + y_pred
        counts = np.bincount(indices, minlength=num_classes**2)
        # Reshape and add to our initialized matrix
        cm += counts.reshape(num_classes, num_classes).astype(cm.dtype)

    # Apply Normalization
    if normalize == 'true':
        sums = cm.sum(axis=1, keepdims=True)
        cm /= np.maximum(sums, 1e-15)
    elif normalize == 'pred':
        sums = cm.sum(axis=0, keepdims=True)
        cm /= np.maximum(sums, 1e-15)
    elif normalize == 'all':
        total = cm.sum()
        cm /= np.maximum(total, 1e-15)

    return cm