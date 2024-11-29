import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class MaskedImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Liste des images
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        # Correspondance des masques
        self.valid_pairs = []
        for img in self.images:
            # Enlever l'extension .jpg et chercher un masque correspondant
            base_name = os.path.splitext(img)[0]
            
            # Trouver le masque qui correspond
            matching_masks = [m for m in os.listdir(mask_dir) if base_name in m and m.endswith('_mask.jpg')]
            
            if matching_masks:
                mask = matching_masks[0]
                self.valid_pairs.append((
                    os.path.join(self.image_dir, img),
                    os.path.join(self.mask_dir, mask)
                ))
            else:
                print(f"Pas de masque trouvé pour {img}")

        if not self.valid_pairs:
            raise ValueError("Aucune correspondance trouvée entre images et masques")
        
        self.transform = transform or transform
        self.transform_mask = transform_mask or transform_mask

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            raise IOError(f"Erreur lors de l'ouverture de l'image ou du masque: {e}")
        
        image = transform(image)
        mask = transform_mask(mask)
        
        return image, mask