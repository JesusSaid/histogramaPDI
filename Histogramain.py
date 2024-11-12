import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para calcular la media global
def calcular_media_global(imagen):
    return np.mean(imagen)

# Función para calcular la varianza global
def calcular_varianza_global(imagen):
    return np.var(imagen)

# Función para calcular la media local
def calcular_media_local(imagen, tamaño_bloque):
    h, w = imagen.shape
    medias_locales = np.zeros_like(imagen, dtype=np.float32)
    
    for i in range(0, h, tamaño_bloque):
        for j in range(0, w, tamaño_bloque):
            bloque = imagen[i:i+tamaño_bloque, j:j+tamaño_bloque]
            media_local = np.mean(bloque)
            medias_locales[i:i+tamaño_bloque, j:j+tamaño_bloque] = media_local
    
    return medias_locales

# Función para calcular la varianza local
def calcular_varianza_local(imagen, tamaño_bloque):
    h, w = imagen.shape
    varianzas_locales = np.zeros_like(imagen, dtype=np.float32)
    
    for i in range(0, h, tamaño_bloque):
        for j in range(0, w, tamaño_bloque):
            bloque = imagen[i:i+tamaño_bloque, j:j+tamaño_bloque]
            varianza_local = np.var(bloque)
            varianzas_locales[i:i+tamaño_bloque, j:j+tamaño_bloque] = varianza_local
    
    return varianzas_locales

# Función para ecualización global
def ecualizar_histograma_global(imagen):
    return cv2.equalizeHist(imagen)

# Función para ecualización local usando CLAHE
def ecualizar_histograma_local(imagen):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(imagen)

# Menú para seleccionar el tipo de ecualización
def seleccionar_histograma():
    print("Seleccione el histograma:")
    print("1.- Global")
    print("2.- Local")
    opcion_histograma = input("Ingrese el número de su elección: ")
    return opcion_histograma

# Menú para seleccionar el tipo de imagen
def seleccionar_imagen():
    print("\nSeleccione la imagen:")
    print("1.- Alto contraste")
    print("2.- Bajo contraste")
    print("3.- Alta iluminación")
    print("4.- Baja iluminación")
    opcion_imagen = input("Ingrese el número de su elección: ")

    # Selección de archivo de imagen según la opción
    if opcion_imagen == "1":
        return 'contrasteAlto.jpg'
    elif opcion_imagen == "2":
        return 'bajoContraste.jpg'
    elif opcion_imagen == "3":
        return 'altaIluminacion.jpg'
    elif opcion_imagen == "4":
        return 'bajaIluminacion.jpg'
    else:
        print("Opción no válida. Seleccionando imagen de alto contraste por defecto.")
        return 'contrasteAlto.jpg'

# Función principal para cargar, procesar y mostrar la imagen
def procesar_imagen():
    # Seleccionar tipo de ecualización y tipo de imagen
    opcion_histograma = seleccionar_histograma()
    ruta_imagen = seleccionar_imagen()
    
    # Cargar la imagen en escala de grises
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    if imagen is None:
        print("Error al cargar la imagen. Verifique el archivo.")
        return

    # Cálculo de media y varianza global
    media_global = calcular_media_global(imagen)
    varianza_global = calcular_varianza_global(imagen)

    # Cálculo de media y varianza local con un bloque de tamaño 8x8
    tamaño_bloque = 8
    media_local = calcular_media_local(imagen, tamaño_bloque)
    varianza_local = calcular_varianza_local(imagen, tamaño_bloque)

    # Imprimir los resultados de media y varianza
    print(f"\nMedia global: {media_global}")
    print(f"Varianza global: {varianza_global}")
    print(f"Media local (promedio de bloque de 8x8): {np.mean(media_local)}")
    print(f"Varianza local (promedio de bloque de 8x8): {np.mean(varianza_local)}")

    # Aplicar la ecualización seleccionada
    if opcion_histograma == "1":
        imagen_ecualizada = ecualizar_histograma_global(imagen)
        titulo_ecualizacion = "Imagen Ecualizada Global"
    elif opcion_histograma == "2":
        imagen_ecualizada = ecualizar_histograma_local(imagen)
        titulo_ecualizacion = "Imagen Ecualizada Local (CLAHE)"
    else:
        print("Opción no válida de histograma. Aplicando ecualización global por defecto.")
        imagen_ecualizada = ecualizar_histograma_global(imagen)
        titulo_ecualizacion = "Imagen Ecualizada Global"

    # Mostrar la imagen original y la ecualizada
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_ecualizada, cmap='gray')
    plt.title(titulo_ecualizacion)
    plt.axis('off')
    
    plt.show()

# Ejecutar el programa
procesar_imagen()
