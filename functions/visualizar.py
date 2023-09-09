from clustimage import Clustimage
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

images_folder = "./input/flower_images/"


# defino funciones para graficos exploratorios

def miro_por_canales(img):
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
    for i in range(3):
        imagen = np.zeros_like(img)  # armo matriz con ceros
        imagen[:, :, i] = img[:, :, i]  # selecciono los canales de a uno
        axs[0, i].imshow(imagen)
        axs[0, i].set_title("RGB"[i])
        axs[0, i].axis('off')

    for i in range(3):
        imagen = np.zeros_like(img)  # armo matriz con ceros
        imagen[:, :, :] = 255        # paso a 255 todos los valores
        imagen[:, :, i] = img[:, :, i]  # selecciono los canales de a uno
        axs[1, i].imshow(imagen)
        axs[1, i].set_title("CMY"[i])
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def separo_y_junto_canales(img):
    fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(15, 8))
    for i in range(3):
        imagen = np.zeros_like(img)  # paso a cero toos los valores
        imagen = img[:, :, i]   # selecciono los canales de a uno
        axs[i].imshow(imagen, cmap='gray')
        axs[i].set_title("RGB"[i])
        axs[i].axis('off')

    suma_capas = (img[:, :, 0]/3 + img[:, :, 1]/3 + img[:, :, 2]/3)

    axs[3].imshow(suma_capas, cmap='gray')
    axs[3].set_title("Suma")
    axs[3].axis('off')

    axs[4].imshow(img)
    axs[4].set_title("Original")
    axs[4].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return None

def tomar_muestra(label_names, muestra=4):
    N = label_names.name.nunique()
    M = muestra
    flowers_sample = (label_names.groupby('label')
                      .sample(n=M, replace=False)
                      .reset_index(drop=True)[['file', 'name']]
                      .set_index('file')
                      .to_records()
                      .reshape(N, M)
                      )
    return flowers_sample

def muestreo_categorias(label_names, muestra=4, images_folder=images_folder):
    N = label_names.name.nunique()
    M = muestra
    flowers_sample = tomar_muestra(label_names=label_names, muestra=M)

    fig, axs = plt.subplots(ncols=M, nrows=N, figsize=(M*4, N*4))
    for i in range(N):
        for j in range(M):
            x = flowers_sample[i, j]
            path = images_folder+x['file']
            name = x['name']
            img = cv2.imread(filename=path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(img)
            axs[i, j].set_title(name)
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return None

def reduce_color(image_array, n_clusters): 
    # Paso a BGR a RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Aplana el array tridimensional a una matriz bidimensional (128*128, 3)
    flattened_image = image_array.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(flattened_image)
    centroides = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Imputo centroide para cada pixel. 
    X_reduced = np.array([centroides[i] for i in labels])

    # Devuelvo en tridimensional
    return X_reduced.reshape(128,128,3)  


def img_to_numpy(path): 
    img = cv2.imread(filename=path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img