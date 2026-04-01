def value_iteration_step(values, transitions, rewards, gamma):
    n_states = len(values)
    new_values = []

    for s in range(n_states):
        best = float('-inf')
        for a in range(len(transitions[s])):
            q = rewards[s][a] + gamma * sum(
                transitions[s][a][s2] * values[s2]
                for s2 in range(n_states)
            )
            if q > best:
                best = q
        new_values.append(best)

    return new_values