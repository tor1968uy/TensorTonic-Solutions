import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    q_values = np.asarray(q_values)
    
    if rng is not None:
        rand = rng.random()
        random_action = int(rng.integers(len(q_values)))
    else:
        rand = np.random.random()
        random_action = int(np.random.randint(len(q_values)))
    
    if rand < epsilon:
        return random_action
    else:
        return int(np.argmax(q_values))