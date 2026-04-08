def promote_model(models):
    """
    Decide qué versión del modelo promover a producción basado en
    exactitud, latencia y fecha.
    """
    
    # Definimos la lógica de ordenamiento:
    # 1. Accuracy (descendente: -x['accuracy'])
    # 2. Latency (ascendente: x['latency'])
    # 3. Timestamp (descendente: x['timestamp'] as string)
    #
    # Al ordenar y tomar el primero (index 0), obtenemos al ganador.
    
    best_model = min(models, key=lambda x: (
        -x['accuracy'],    # El signo menos convierte el mayor valor en el "mínimo"
        x['latency'],      # Queremos el valor más bajo
        -int(x['timestamp'].replace('-', '')) # El más reciente es el número más alto
    ))
    
    # Alternativa usando sorted():
    # sorted_models = sorted(models, key=lambda x: (x['accuracy'], -x['latency'], x['timestamp']), reverse=True)
    # return sorted_models[0]['name']
    
    return best_model['name']