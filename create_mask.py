import os
import cv2
import numpy as np

# Chemin vers le dossier contenant les sous-dossiers
input_dir = "/Users/yannischappetjuan/Desktop/IA/ClassifiorGAN/output/subset_clear_1"
output_dir = "/Users/yannischappetjuan/Desktop/IA/ClassifiorGAN/output/MASK_SUBSET_1"

# Créer le dossier MASK s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Fonction pour traiter une image et générer son masque
def process_image(image_path):
    # 1) Chargement de l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 2) Pré-traitement : flou gaussien
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 3) Seuillage (méthode d'Otsu)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    # 4) Transformations morphologiques
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5) Détection du plus grand contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6) Création du masque final
    final_mask = np.zeros_like(closed)
    if len(contours) > 0:
        contour_areas = [cv2.contourArea(c) for c in contours]
        max_idx = np.argmax(contour_areas)
        largest_contour = contours[max_idx]
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return final_mask

# Parcourir les sous-dossiers et traiter les images
for subdir, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):  # Vérifier l'extension des fichiers
            # Chemin complet de l'image
            image_path = os.path.join(subdir, file)

            # Générer le masque
            mask = process_image(image_path)

            if mask is not None:
                # Construire le chemin de sauvegarde correspondant dans le dossier MASK
                relative_path = os.path.relpath(subdir, input_dir)
                mask_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(mask_subdir, exist_ok=True)
                output_path = os.path.join(mask_subdir, file)

                # Sauvegarder le masque
                cv2.imwrite(output_path, mask)

print("Traitement terminé. Les masques sont enregistrés dans le dossier:", output_dir)
