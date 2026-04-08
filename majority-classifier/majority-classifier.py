import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    Returns: NumPy array of predictions.
    """
    # 1. Handle edge case: empty training data
    if len(y_train) == 0:
        return np.array([], dtype=int)
        
    # 2. Find the most frequent class
    # return_counts=True gives us the unique labels and their respective frequencies
    classes, counts = np.unique(y_train, return_counts=True)
    
    # 3. Find the index of the maximum count
    # np.argmax returns the first index in case of a tie, ensuring stable behavior
    majority_class_index = np.argmax(counts)
    majority_class = classes[majority_class_index]
    
    # 4. Generate predictions for the test set
    # The output should have the same number of rows as X_test
    num_test_samples = len(X_test)
    predictions = np.full(shape=(num_test_samples,), fill_value=majority_class, dtype=int)
    
    return predictions