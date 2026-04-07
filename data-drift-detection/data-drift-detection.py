def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    Returns: dict with "score" (float) and "drift_detected" (bool).
    """
    # 1. Normalize reference counts to probability distribution p
    total_ref = sum(reference_counts)
    p = [count / total_ref for count in reference_counts]
    
    # 2. Normalize production counts to probability distribution q
    total_prod = sum(production_counts)
    q = [count / total_prod for count in production_counts]
    
    # 3. Calculate Total Variation Distance (TVD)
    # TVD = 0.5 * sum(|p_i - q_i|)
    total_abs_diff = 0.0
    for i in range(len(p)):
        total_abs_diff += abs(p[i] - q[i])
        
    score = 0.5 * total_abs_diff
    
    # 4. Check if drift is strictly greater than the threshold
    drift_detected = score > threshold
    
    return {
        "score": float(score),
        "drift_detected": bool(drift_detected)
    }