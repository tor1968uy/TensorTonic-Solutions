import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Estimate the value function using Monte Carlo first-visit returns.
    Returns: V (NumPy array of shape (n_states,))
    """
    # 1. Initialize accumulators for sums and counts
    sum_returns = np.zeros(n_states, dtype=float)
    count_returns = np.zeros(n_states, dtype=float)
    
    for episode in episodes:
        # 2. Prepare to calculate returns backward
        g = 0.0
        visited_in_episode = set()
        # Store returns for the first-visit of each state in this episode
        first_visit_returns = {}
        
        # 3. Iterate backward: t = T-1, T-2, ..., 0
        for i in range(len(episode) - 1, -1, -1):
            state, reward = episode[i]
            # Update return: G_t = r_t + gamma * G_{t+1}
            g = reward + gamma * g
            
            # Since we are going backward, the 'last' time we see a state 
            # in this loop is actually its 'first' visit in chronological order.
            first_visit_returns[state] = g
            
        # 4. Accumulate the first-visit returns found in this episode
        for state, g_return in first_visit_returns.items():
            sum_returns[state] += g_return
            count_returns[state] += 1
            
    # 5. Compute the average: V(s) = Sum(G) / Count
    # Use np.divide to handle states with 0 visits safely
    v = np.zeros(n_states)
    mask = count_returns > 0
    v[mask] = sum_returns[mask] / count_returns[mask]
    
    return v