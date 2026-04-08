import math

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    Returns: float - the calculated learning rate for the current step.
    """
    # 1. Calculate the progress through the total training period (0.0 to 1.0)
    # The cosine argument moves from 0 to pi (3.1415...)
    progress = current_step / total_steps
    
    # 2. Compute the cosine component: cos(pi * current_step / total_steps)
    # At start: cos(0) = 1.0
    # At end: cos(pi) = -1.0
    cos_val = math.cos(math.pi * progress)
    
    # 3. Apply the scaling formula
    # (1 + cos_val) ranges from 2.0 down to 0.0
    # Multiplying by 0.5 makes it range from 1.0 down to 0.0
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos_val)
    
    return float(lr)