def policy_gradient_loss(log_probs, rewards, gamma):
    T = len(rewards)
    
    # Calcular retornos hacia atrás
    G = [0.0] * T
    G[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    
    mean_G = sum(G) / T
    
    advantages = [g - mean_G for g in G]
    
    loss = -sum(lp * a for lp, a in zip(log_probs, advantages)) / T
    
    return loss