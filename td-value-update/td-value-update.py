import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Perform a one-step TD(0) update for a state value function.
    Returns: V_new (NumPy array)
    """
    # 1. Create a copy to ensure we don't modify the original array in-place
    V_new = np.array(V, copy=True, dtype=float)
    
    # 2. Calculate the TD Target
    # Target = Immediate Reward + Discounted Value of Next State
    td_target = r + gamma * V_new[s_next]
    
    # 3. Calculate the TD Error (delta)
    # How far off was our current estimate V[s] from the target?
    td_error = td_target - V_new[s]
    
    # 4. Perform the update
    # New Value = Old Value + Learning Rate * TD Error
    V_new[s] = V_new[s] + alpha * td_error
    
    return V_new