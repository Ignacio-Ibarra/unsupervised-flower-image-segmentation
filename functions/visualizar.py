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

def muestreo_categorias_inicial(label_names, images_folder=images_folder, save_path=None):
    
    flowers_sample = tomar_muestra(label_names=label_names, muestra=1).reshape(2,5)

    fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(20,8))
    for i in range(2):
        for j in range(5):
            x = flowers_sample[i, j]
            path = images_folder+x['file']
            name = x['name']
            img = cv2.imread(filename=path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(img)
            axs[i, j].set_title(name)
            axs[i, j].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)
    return None

def muestreo_categorias(label_names, muestra=4, images_folder=images_folder, save_path=None):
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
    if save_path:
        plt.savefig(save_path)
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

def divido_y_mezclo(imagen_color, divisiones):
    if divisiones not in [1, 2, 4, 8, 16, 32, 64, 128]:
        print('Solo se aceptan como valores los divisores enteros de 128')
        print('1, 2, 4, 8, 16, 32, 64 o 128')
        return None
        
    imagen_color2 = imagen_color.copy()
    lista_partes=[]
    tamaño_parte = int(128/divisiones)
    for i in range(divisiones):
        for j in range(divisiones):
            x = i*tamaño_parte
            y = j*tamaño_parte
            lista_partes.append(imagen_color2[x:x+tamaño_parte, y:y+tamaño_parte, :].copy()) #faltaba el copy y entonces reasignaba de manera loca

    random.shuffle(lista_partes)
    select_lista = 0
    
    for i in range(divisiones):
        for j in range(divisiones):
            x = i*tamaño_parte
            y = j*tamaño_parte
            
            imagen_color2[x:x+tamaño_parte, y:y+tamaño_parte, :] = lista_partes[select_lista]
            select_lista = select_lista + 1
    
    plt.imshow(imagen_color2)
    return None


def divido_dos_y_mezclo(imagen_color, imagen_dos, divisiones):
    if divisiones not in [1, 2, 4, 8, 16, 32, 64, 128]:
        print('Solo se aceptan como valores los divisores enteros de 128')
        print('1, 2, 4, 8, 16, 32, 64 o 128')
        return None
        
    imagen_color2 = imagen_color.copy()
    lista_partes=[]
    tamaño_parte = int(128/divisiones)
    for i in range(divisiones):
        for j in range(divisiones):
            x = i*tamaño_parte
            y = j*tamaño_parte
            lista_partes.append(imagen_color2[x:x+tamaño_parte, y:y+tamaño_parte, :].copy()) #faltaba el copy y entonces reasignaba de manera loca
    
    imagen_color3 = imagen_dos.copy()
    lista_partes_dos=[]
    tamaño_parte = int(128/divisiones)
    for i in range(divisiones):
        for j in range(divisiones):
            x = i*tamaño_parte
            y = j*tamaño_parte
            lista_partes_dos.append(imagen_color3[x:x+tamaño_parte, y:y+tamaño_parte, :].copy()) #faltaba el copy y entonces reasignaba de manera loca

    
    lista_uno = random.sample(lista_partes, int((divisiones*divisiones)/2))
    lista_dos = random.sample(lista_partes_dos, int((divisiones*divisiones)/2))
    lista_combinada = lista_uno + lista_dos
    random.shuffle(lista_combinada)
    select_lista = 0
    
    for i in range(divisiones):
        for j in range(divisiones):
            x = i*tamaño_parte
            y = j*tamaño_parte
            
            imagen_color2[x:x+tamaño_parte, y:y+tamaño_parte, :] = lista_combinada[select_lista]
            select_lista = select_lista + 1
    
    plt.imshow(imagen_color2)
    return None

def change_brightness(img, value=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img