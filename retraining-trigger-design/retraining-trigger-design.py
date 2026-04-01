def retraining_policy(daily_stats, config):
    """
    Decide qué días activar el reentrenamiento basándose en drift, performance y staleness.
    """
    drift_thresh = config["drift_threshold"]
    perf_thresh = config["performance_threshold"]
    max_staleness = config["max_staleness"]
    cooldown = config["cooldown"]
    cost = config["retrain_cost"]
    budget = config["budget"]

    retrain_days = []
    # Estado inicial: el modelo empieza "fresco"
    # Usamos None o un valor que no dispare staleness el día 1
    last_retrain_day = None 
    current_budget = budget

    for stats in daily_stats:
        day = stats["day"]
        drift = stats["drift_score"]
        perf = stats["performance"]
        
        # Calcular staleness: días que han pasado desde el inicio o el último reentrenamiento
        # Si no hubo reentrenamiento, contamos desde el día 0 (o simplemente day)
        if last_retrain_day is None:
            days_since = day
        else:
            days_since = day - last_retrain_day

        # 1. Condiciones de activación (Triggers)
        trigger_drift = drift > drift_thresh
        trigger_perf = perf < perf_thresh
        trigger_staleness = days_since >= max_staleness
        
        should_retrain = trigger_drift or trigger_perf or trigger_staleness

        # 2. Restricciones (Constraints)
        can_afford = current_budget >= cost
        
        # Cooldown: si es la primera vez (None), siempre cumple. 
        # Si no, la diferencia de días debe ser >= cooldown.
        if last_retrain_day is None:
            meets_cooldown = True
        else:
            meets_cooldown = (day - last_retrain_day) >= cooldown

        # 3. Ejecución
        if should_retrain and meets_cooldown and can_afford:
            retrain_days.append(day)
            last_retrain_day = day
            current_budget -= cost

    return retrain_days

# --- Validación del Caso 1 ---
config1 = {"budget":500,"cooldown":1,"retrain_cost":100,"max_staleness":30,"drift_threshold":0.5,"performance_threshold":0.7}
stats1 = [
    {"day":1,"drift_score":0.1,"performance":0.95},
    {"day":2,"drift_score":0.3,"performance":0.93},
    {"day":3,"drift_score":0.6,"performance":0.9},
    {"day":4,"drift_score":0.2,"performance":0.94}
]
# Ahora devolverá [3] porque el día 1 (drift 0.1, perf 0.95, staleness 1) no activa triggers.
print(f"Resultado: {retraining_policy(stats1, config1)}")