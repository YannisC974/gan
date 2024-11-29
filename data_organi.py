import os
import shutil

# Organise les images de chaque œil dans trois dossiers distincts (CNV, DRUSEN, NORMAL), 
# sans tenir compte de la classe d'origine du patient. 
# Le code crée automatiquement les dossiers nécessaires dans le répertoire de destination (./sorted_images),
#  parcourt les sous-dossiers de chaque catégorie de départ et copie les images correspondantes dans 
# le bon dossier en renommant chaque fichier de manière unique selon son emplacement d'origine.

base_dir = './' 
input_categories = ['CNV', 'DRUSEN', 'NORMAL']
output_dir = './sorted_images'  # Dossier de destination pour les images triées

# Créer les dossiers de destination pour chaque catégorie
for category in input_categories:
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

# Parcourir les fichiers dans les sous-dossiers de chaque catégorie
for input_category in input_categories:
    category_dir = os.path.join(base_dir, input_category)
    
    for root, _, files in os.walk(category_dir):
        for file in files:
            if file.endswith(('.jpg', '.tif')):
                target_category = None
                if 'CNV' in file.upper():
                    target_category = 'CNV'
                elif 'DRUSEN' in file.upper():
                    target_category = 'DRUSEN'
                elif 'NORMAL' in file.upper():
                    target_category = 'NORMAL'
                
                if target_category:
                    relative_path = os.path.relpath(root, category_dir)
                    new_name = f"{relative_path.replace(os.sep, '_')}_{file}"
                    
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(output_dir, target_category, new_name)
                    shutil.copy2(src_path, dest_path)
                    print(f"Copié : {src_path} -> {dest_path}")
