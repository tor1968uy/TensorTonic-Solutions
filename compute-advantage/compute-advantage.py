import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Compute the advantage for every time step in an episode.
    Returns: A (NumPy array of advantages)
    """
    n = len(rewards)
    returns = np.zeros(n, dtype=float)
    
    # 1. Compute returns backward: G_t = r_t + gamma * G_{t+1}
    # Start from the last reward (T-1)
    current_g = 0
    for t in range(n - 1, -1, -1):
        current_g = rewards[t] + gamma * current_g
        returns[t] = current_g
    
    # 2. Extract state values for each state visited in the episode
    # states[t] gives the index to look up in the V array
    state_values = np.array([V[s] for s in states])
    
    # 3. Advantage = Return - Expected Value
    # A_t = G_t - V(s_t)
    advantages = returns - state_values
    
    return advantages