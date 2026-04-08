def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    Returns: A list of merged feature dictionaries.
    """
    combined_vectors = []
    
    for request in requests:
        user_id = request["user_id"]
        online_data = request["online_features"]
        
        # 1. Look up offline features in the store
        # If the user_id is missing, use the 'defaults' dictionary
        offline_data = feature_store.get(user_id, defaults)
        
        # 2. Merge the two dictionaries
        # The {**d1, **d2} syntax creates a new dict with key-value pairs from both
        merged_features = {**offline_data, **online_data}
        
        combined_vectors.append(merged_features)
        
    return combined_vectors