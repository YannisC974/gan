import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset
import shutil
from torchvision.transforms.functional import to_pil_image

# Dictionnaire des labels pour les différentes classes
label_map = {
    "CNV": 0,
    "DRUSEN": 1,
    "NORMAL": 2
}

# Classe Dataset pour la classification d'images
class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Le répertoire racine contenant les images.
            transform (callable, optional): Transformation à appliquer sur les images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Parcours du répertoire racine et des sous-répertoires pour récupérer les images
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # Vérifie que le fichier est une image (par extension)
                if file.endswith(('.jpg', '.tif')):
                    # Construction du chemin complet de l'image
                    file_path = os.path.join(root, file)

                    # Le label est basé sur le nom du dossier principal
                    if 'CNV' in file:
                        label = label_map["CNV"]
                    elif 'Drusen' in file:
                        label = label_map["DRUSEN"]
                    elif 'Normal' in file:
                        label = label_map["NORMAL"]
                    else:
                        continue

                    # Ajoute le chemin de l'image et son label
                    self.image_paths.append(file_path)
                    self.labels.append(label)
                    # print(file_path, label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Ouvre l'image et la convertit en RGB

        label = self.labels[idx]  # Récupère le label correspondant à l'image

        if self.transform:
            image = self.transform(image)

        return image, label

class EyeDatasetOutput(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Le répertoire racine contenant les sous-dossiers par classe.
            transform (callable, optional): Transformation à appliquer sur les images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Parcours des sous-dossiers et des fichiers
        for class_name, label in label_map.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise ValueError(f"Le dossier {class_dir} est introuvable.")
            
            for file in os.listdir(class_dir):
                # Vérifie que le fichier est une image (par extension)
                if file.endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    file_path = os.path.join(class_dir, file)
                    self.image_paths.append(file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Récupération du chemin de l'image et du label correspondant
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Chargement de l'image
        image = Image.open(img_path).convert("L")  # Convertit en RGB

        # Application des transformations (si spécifiées)
        if self.transform:
            image = self.transform(image)

        return image, label

def split_dataset_balanced(dataset, seed=42):
    """
    Sépare un dataset en deux sous-ensembles équilibrés en fonction des labels (0, 1, 2).
    
    Args:
        dataset (torch.utils.data.Dataset): Le dataset à diviser.
        seed (int): La seed pour garantir la reproductibilité (par défaut 42).
        
    Returns:
        subset1 (torch.utils.data.Subset): Premier sous-ensemble équilibré.
        subset2 (torch.utils.data.Subset): Deuxième sous-ensemble équilibré.
    """
    np.random.seed(seed)  # Fixer la seed pour la reproductibilité
    
    # Créer un dictionnaire pour stocker les indices par classe
    class_indices = defaultdict(list)
    
    # Parcourir le dataset et ajouter les indices aux classes respectives
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Créer les indices pour les deux sous-ensembles équilibrés
    subset1_indices = []
    subset2_indices = []
    
    # Répartir les indices de chaque classe dans les deux sous-ensembles
    for label, indices in class_indices.items():
        print(indices)
        # Mélanger les indices pour chaque classe
        np.random.shuffle(indices)
        
        # Séparer en deux parties égales et les répartir entre subset1 et subset2
        mid = len(indices) // 2
        subset1_indices.extend(indices[:mid])
        subset2_indices.extend(indices[mid:])
    
    # Créer les sous-ensembles en utilisant Subset
    subset1 = Subset(dataset, subset1_indices)
    subset2 = Subset(dataset, subset2_indices)
    
    return subset1, subset2

def save_split_to_folders(subset1, subset2, output_dir):
    """
    Sauvegarde les sous-ensembles dans des dossiers organisés par classes.

    Args:
        subset1 (torch.utils.data.Subset): Premier sous-ensemble.
        subset2 (torch.utils.data.Subset): Deuxième sous-ensemble.
        output_dir (str): Répertoire de sortie pour sauvegarder les données.
    """
    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Définir les répertoires pour subset1 et subset2
    subset1_dir = os.path.join(output_dir, "subset1")
    subset2_dir = os.path.join(output_dir, "subset2")

    # Supprimer les répertoires existants pour éviter les conflits
    shutil.rmtree(subset1_dir, ignore_errors=True)
    shutil.rmtree(subset2_dir, ignore_errors=True)

    # Crée les répertoires pour subset1 et subset2
    os.makedirs(subset1_dir, exist_ok=True)
    os.makedirs(subset2_dir, exist_ok=True)

    # Dictionnaire inverse pour mapper les labels aux noms des classes
    class_map = {v: k for k, v in label_map.items()}

    def save_images(subset, base_dir):
        """Sauvegarde les images d'un subset dans un répertoire."""
        for img, label in subset:
            # Convertir le tenseur en image PIL
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img)

            # Nom de la classe
            class_name = class_map[label]
            class_dir = os.path.join(base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Générer un nom de fichier unique
            img_name = f"{class_name}_{len(os.listdir(class_dir)) + 1}.jpg"

            # Chemin complet pour l'image
            img_path = os.path.join(class_dir, img_name)

            # Sauvegarder l'image
            img.save(img_path)

    # Sauvegarder les données de subset1 et subset2
    save_images(subset1, subset1_dir)
    save_images(subset2, subset2_dir)