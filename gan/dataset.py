import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class Dataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Charger les chemins des fichiers à partir des 3 sous-dossiers
        self.image_paths = []
        for subfolder in self.root_dir.iterdir():
            if subfolder.is_dir():
                self.image_paths.extend(list(subfolder.glob("*.jpg")))

        if not self.image_paths:
            raise ValueError(f"Aucune image trouvée dans {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image