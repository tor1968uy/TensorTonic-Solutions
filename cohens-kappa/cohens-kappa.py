import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient between two raters.
    Returns: float - the kappa score.
    """
    n = len(rater1)
    if n == 0:
        return 0.0
    
    # 1. Convert to NumPy arrays for efficient processing
    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)
    
    # 2. Observed Agreement (p_o)
    # The fraction of cases where both raters gave the same label
    agreements = np.sum(r1 == r2)
    p_o = agreements / n
    
    # 3. Expected Agreement (p_e)
    # Identify all unique labels present across both raters
    labels = np.unique(np.concatenate([r1, r2]))
    p_e = 0.0
    
    for label in labels:
        # Probability that rater 1 picks this label by chance
        count1 = np.sum(r1 == label)
        prob1 = count1 / n
        
        # Probability that rater 2 picks this label by chance
        count2 = np.sum(r2 == label)
        prob2 = count2 / n
        
        # Expected probability of both picking this label simultaneously
        p_e += (prob1 * prob2)
        
    # 4. Compute Kappa
    # Handle the degenerate case where perfect agreement is expected by chance (p_e = 1)
    if p_e == 1.0:
        return 1.0
        
    kappa = (p_o - p_e) / (1 - p_e)
    
    return float(kappa)