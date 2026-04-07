def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order).
    This implementation is case-sensitive as per requirements.
    """
    # 1. Convert stopword list to a set for O(1) average-time complexity lookups
    stop_set = set(stopwords)
    
    # 2. Use list comprehension to filter the tokens
    # We only keep 'token' if it is NOT in our set of stopwords
    filtered_tokens = [token for token in tokens if token not in stop_set]
    
    return filtered_tokens