def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    Returns: list of dicts.
    """
    # 1. Track the 'best' record for each unique key
    # and the order in which each key was first seen.
    best_records = {}
    first_appearance_order = []
    
    for record in records:
        # 2. Create a unique key (tuple) from the specified key_columns
        # Using a tuple allows for composite keys (multiple columns)
        key = tuple(record[col] for col in key_columns)
        
        # 3. If this is the first time we see this key, record its appearance order
        if key not in best_records:
            first_appearance_order.append(key)
            best_records[key] = record
            continue
        
        # 4. If we've seen the key before, apply the selection strategy
        if strategy == "last":
            # "last" simply overwrites the previous one with the current one
            best_records[key] = record
            
        elif strategy == "most_complete":
            # Count None values in both the current 'best' and the new candidate
            # The record with the FEWEST None values wins
            current_best_none_count = sum(1 for v in best_records[key].values() if v is None)
            new_candidate_none_count = sum(1 for v in record.values() if v is None)
            
            if new_candidate_none_count < current_best_none_count:
                best_records[key] = record
        
        # Strategy "first" requires no action if the key is already in best_records
            
    # 5. Build the final list based on the order keys first appeared
    return [best_records[key] for key in first_appearance_order]