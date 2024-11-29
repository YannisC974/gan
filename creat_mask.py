import numpy as np
import cv2
import random
import os
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

input_dir = './../sorted_images/'
output_dir = './../sorted_images/mask/'
mask_dir = os.path.join(output_dir, 'mask/')
mask_transform_dir = os.path.join(output_dir, 'mask_transform/')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(mask_transform_dir, exist_ok=True)


### Function pour creer un masque binaire ###


# Retirer l'échelle de l'image
def image_without_scale(image):
    image[420: , :60] = 0 # la position de l'echelle
    return image


# Rendre l'image d'origine binaire (seuillage)
def threshold_mask(image):
    blurred_image = cv2.GaussianBlur(image, (5,5), 0) 
    _, mask = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# Methode 1 : Lisser les bordures du masque binaire
def fill_and_smooth_boundaries(mask, cutoff_threshold=0.8):
    height, width = mask.shape
    
    # Initialisation des frontières supérieure et inférieure
    upper_border = np.zeros(width, dtype=int)
    lower_border = np.zeros(width, dtype=int)

    # Détection des frontières supérieure et inférieure
    for x in range(width):
        column = mask[:, x]
        white_pixels = np.where(column == 255)[0]
        if len(white_pixels) > 0:
            upper_border[x] = white_pixels[0]
            lower_border[x] = white_pixels[-1]
        else:
            upper_border[x] = 0
            lower_border[x] = height - 1

    # Calcul de la largeur de chaque colonne blanche
    column_widths = lower_border - upper_border
    max_width = height * cutoff_threshold  # Seuil pour couper les extrémités
    valid_columns = np.where(column_widths < max_width)[0]  # Colonnes à conserver

    # Lissage des frontières
    smoothed_upper_border = gaussian_filter1d(upper_border, sigma=10)
    smoothed_lower_border = gaussian_filter1d(lower_border, sigma=10)

    # Limitation aux colonnes valides
    min_col, max_col = valid_columns[0], valid_columns[-1]

    # Création d'un masque lissé basé sur les frontières dans la zone centrale
    smoothed_mask = np.zeros_like(mask)
    for x in range(min_col, max_col + 1):
        start = int(smoothed_upper_border[x])
        end = int(smoothed_lower_border[x])
        smoothed_mask[start:end+1, x] = 255  # Remplissage en blanc entre les frontières

    return smoothed_mask[:, min_col:max_col + 1]  # Retourner la zone centrale uniquement

# Methode 2 : Lisser les bordures du masque binaire
def eroded_and_dilated_mask(mask) :
    kernel_close = np.ones((30, 30), np.uint8)  
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel = np.ones((20, 20), np.uint8) 
    eroded_mask = cv2.erode(closed_mask, kernel, iterations=1)
    kernel = np.ones((20, 20), np.uint8) 
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)
    blurred_mask = cv2.GaussianBlur(dilated_mask, (5, 5), 0)
    return blurred_mask


# Generer un masque binaire à partir d'une image OCT
def generate_mask(image): 
    image = image_without_scale(image)
    mask = threshold_mask(image)
    smoothed_mask = fill_and_smooth_boundaries(mask) # Usage de la méthode 1
    return smoothed_mask


### Function pour transformer un masque binaire ###


# Appliquer une transformation sinusoidale
def sinusoidal_mask(mask):
    height, width = mask.shape
    a = random.randint(0, 150)
    d = random.randint(-100, 100)
    diff_Y = np.abs(np.sin(np.linspace(0, np.pi, width)) * (-a)) + d
    deformed_mask = np.zeros_like(mask)
    for x in range(width):
        for y in range(height):
            new_y = int(y + diff_Y[x])
            if 0 <= new_y < height:
                deformed_mask[new_y, x] = mask[y, x]
    return deformed_mask


# Appliquer d'une rotation aléatoire
def rotate_mask(mask):
    angle = random.uniform(-15, 15)
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated_mask


# Transformer un masque binaire 
def transform_mask(mask): 
    transform_mask = sinusoidal_mask(mask)
    transform_mask = rotate_mask(transform_mask)
    return transform_mask


### Function pour afficher les masques binaires ###


def display_images(images, titles, cols=3):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, rows * 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()


if __name__ == "__main__":

    for categorie in os.listdir(input_dir):

        if categorie != 'mask':
        
            input_dir = os.path.join(input_dir, categorie)
            output_dir = os.path.join(output_dir, categorie)

            for filename in os.listdir(input_dir):

                if ( filename.endswith(('.jpg','.tif'))):

                    # Chargement de l'image d'entrée
                    image_path = os.path.join(input_dir, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    mask = generate_mask(image)

                    mask_filename = filename.replace('.jpg', '_mask.jpg').replace('.tif', '_mask.tif')
                    mask_path = os.path.join(mask_dir, mask_filename)
                    cv2.imwrite(mask_path, mask)
                    
                    # Application d'une transformation
                    deformed_mask = transform_mask(mask)
                    
                    mask_transform_filename = filename.replace('.jpg', '_mask_transform.jpg').replace('.tif', '_mask_transform.tif')
                    mask_transform_path = os.path.join(mask_transform_dir, mask_transform_filename)
                    cv2.imwrite(mask_transform_path, deformed_mask)

    print("Création des masques terminée.")


