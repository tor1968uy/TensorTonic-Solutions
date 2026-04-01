def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        # Warmup: 0 → initial_lr
        return initial_lr * step / warmup_steps
    elif step >= total_steps:
        # Clamped
        return float(final_lr)
    else:
        # Decay: initial_lr → final_lr
        decay_steps = total_steps - warmup_steps
        decay_progress = (step - warmup_steps) / decay_steps
        return initial_lr + (final_lr - initial_lr) * decay_progress