def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # 1. Crear una lista de 256 ceros (bins del 0 al 255)
    hist = [0] * 256
    
    # 2. Recorrer cada fila de la imagen
    for row in image:
        # 3. Recorrer cada píxel de la fila
        for pixel in row:
            # 4. Incrementar el bin correspondiente al valor del píxel
            hist[pixel] += 1
            
    # 5. Retornar la lista con los 256 conteos
    return hist