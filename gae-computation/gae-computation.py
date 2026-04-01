def gae(rewards, values, gamma, lam):
    T = len(rewards)
    advantages = [0.0] * T
    last_adv = 0.0

    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = last_adv

    return advantages