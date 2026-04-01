def schedule_pipeline(tasks, resource_budget):
    """
    Programa tareas de ETL respetando dependencias y presupuesto de recursos.
    """
    # 1. Preparación de datos
    task_dict = {t['name']: t for t in tasks}
    remaining_tasks = sorted(tasks, key=lambda x: x['name'])
    
    completed_tasks = set()
    running_tasks = [] # Lista de (end_time, task_name, resources)
    scheduled_output = []
    
    current_time = 0
    current_resources = 0
    
    while len(scheduled_output) < len(tasks):
        # A. Finalizar tareas que terminan en current_time o antes
        # (Aunque avanzamos por eventos, este paso limpia el estado)
        still_running = []
        for end_time, name, res in running_tasks:
            if end_time <= current_time:
                completed_tasks.add(name)
                current_resources -= res
            else:
                still_running.append((end_time, name, res))
        running_tasks = still_running

        # B. Intentar iniciar nuevas tareas (respetando orden alfabético)
        # Usamos una lista temporal para las tareas que aún no podemos programar
        not_started = []
        for task in remaining_tasks:
            name = task['name']
            deps = task['depends_on']
            req_res = task['resources']
            
            # Condición 1: Dependencias satisfechas
            deps_met = all(d in completed_tasks for d in deps)
            # Condición 2: Recursos disponibles
            res_avail = (current_resources + req_res) <= resource_budget
            
            if deps_met and res_avail:
                # Iniciar tarea
                start_time = current_time
                end_time = start_time + task['duration']
                
                scheduled_output.append((name, start_time))
                running_tasks.append((end_time, name, req_res))
                current_resources += req_res
            else:
                not_started.append(task)
        
        remaining_tasks = not_started

        # C. Avanzar el tiempo al próximo evento (finalización de una tarea)
        if len(scheduled_output) < len(tasks):
            if running_tasks:
                # El siguiente evento es el tiempo de finalización más cercano
                current_time = min(t[0] for t in running_tasks)
            else:
                # Caso de seguridad: si no hay nada corriendo pero faltan tareas
                # (No debería ocurrir en un DAG válido con recursos suficientes)
                current_time += 1

    # Ordenar el output por tiempo de inicio y luego alfabéticamente
    return sorted(scheduled_output, key=lambda x: (x[1], x[0]))

# --- Validación con ejemplos ---
tasks1 = [
    {"name": "extract", "duration": 2, "resources": 1, "depends_on": []},
    {"name": "transform", "duration": 3, "resources": 1, "depends_on": ["extract"]},
    {"name": "load", "duration": 1, "resources": 1, "depends_on": ["transform"]},
]
print(f"Ejemplo 1 (Lineal): {schedule_pipeline(tasks1, 2)}")

tasks2 = [
    {"name": "fetch_orders", "duration": 3, "resources": 1, "depends_on": []},
    {"name": "fetch_users", "duration": 2, "resources": 1, "depends_on": []},
    {"name": "join", "duration": 1, "resources": 2, "depends_on": ["fetch_users", "fetch_orders"]},
]
print(f"Ejemplo 2 (Paralelo): {schedule_pipeline(tasks2, 2)}")