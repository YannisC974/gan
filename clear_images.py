import os
from PIL import Image, ImageDraw

input_folder = "/Users/yannischappetjuan/Desktop/IA/ClassifiorGAN/output/subset2"
output_folder = "/Users/yannischappetjuan/Desktop/IA/ClassifiorGAN/output/subset_clear_2"

rectangle_width = 21
rectangle_height = 33

os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(".jpg"):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)

            output_path = os.path.join(output_subfolder, file)

            with Image.open(input_path) as img:
                # Convertir en mode RGB pour éviter les artefacts
                img = img.convert("RGB")

                # Taille de l'image
                img_width, img_height = img.size

                # Ajuster les dimensions du rectangle si nécessaire
                rect_width = min(rectangle_width, img_width)
                rect_height = min(rectangle_height, img_height)

                # Coordonnées du rectangle (bas gauche)
                x0, y0 = 0, img_height - rect_height
                x1, y1 = rect_width, img_height

                # Dessiner le rectangle noir
                draw = ImageDraw.Draw(img)
                draw.rectangle([x0, y0, x1, y1], fill="black")

                # Sauvegarder avec qualité ajustée
                img.save(output_path, quality=95)

print("Traitement terminé. Les images modifiées sont dans:", output_folder)

