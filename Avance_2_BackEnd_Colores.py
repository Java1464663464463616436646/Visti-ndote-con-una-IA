import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

def extract_colors(image):
    # Convertir la imagen a un espacio de color adecuado (por ejemplo, RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reformatear la imagen en una matriz 2D de píxeles
    pixels = image_rgb.reshape((-1, 3))
    # Convertir los datos a float32
    pixels = np.float32(pixels)
    # Definir criterios de parada del algoritmo de K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # Número de colores a extraer
    k = 5  # Puedes ajustar este valor según tus necesidades
    # Ejecutar el algoritmo de K-means
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convertir los centros de los clusters a valores enteros
    centers = np.uint8(centers)
    
    return centers

# Crear una ventana emergente para seleccionar la imagen
Tk().withdraw() 
image_path = askopenfilename()  # Abrir una ventana para seleccionar la imagen

if image_path:  # Si se selecciona una imagen
    # Cargar la imagen desde el archivo
    image = cv2.imread(image_path)

    if image is not None:
        # Extraer los colores de la imagen
        colors = extract_colors(image)
        print("Colores extraídos:")
        print(colors)
        
        # Crear una imagen compuesta por los colores dominantes
        color_palette = np.zeros((50, len(colors) * 50, 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            color_palette[:, i*50:(i+1)*50] = color
        plt.imshow(color_palette)
        plt.axis('off')
        plt.title('Colores Dominantes')
        plt.show()
    else:
        print("No se pudo cargar la imagen. Asegúrate de que la ruta del archivo sea correcta.")
else:
    print("No se seleccionó ninguna imagen.")
