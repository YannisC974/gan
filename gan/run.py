import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from GAN import UNet, Discriminator

device = torch.device('mps' if torch.backends.mps.is_available() 
                         else 'cuda' if torch.cuda.is_available() 
                         else 'cpu')

generator = UNet().to(device)

checkpoint_path = "/home/ychappetjuan/free-projetIA12456/logs/run_20241129_160820/checkpoints/checkpoint_200.pth"
checkpoint = torch.load(checkpoint_path)

# Extraire uniquement les poids du générateur
generator_state_dict = checkpoint.get("generator_state_dict")

if generator_state_dict:
    generator.load_state_dict(generator_state_dict)
    generator.eval()
else:
    raise ValueError("Les poids du générateur sont introuvables dans le checkpoint.")


image_path = '/home/ychappetjuan/free-projetIA12456/masks/mask_transform/003_Normal_24_transform_mask.jpg'
image = Image.open(image_path).convert("L")

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dossier contenant les masques
input_folder = '/home/ychappetjuan/free-projetIA12456/masks/mask'
output_folder = '/home/ychappetjuan/free-projetIA12456/generated_images/'

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# input_image = transform_mask(image).unsqueeze(0).to(device)  

# with torch.no_grad():
#     generated_image = generator(input_image)

# generated_image = generated_image[0].detach().cpu() 
# generated_image = (generated_image * 0.5 + 0.5)  
# generated_image = generated_image.permute(1, 2, 0)  
# generated_image = (generated_image * 255).byte()  

# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title("Input Image")
# plt.subplot(1, 2, 2)
# plt.imshow(generated_image, cmap='gray')
# plt.title("Generated Image")
# plt.show()

# Parcourir tous les fichiers du dossier
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Vérifier les formats d'image
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("L")  # Charger le masque
        
        # Appliquer la transformation
        input_image = transform_mask(image).unsqueeze(0).to(device)
        
        # Générer l'image
        with torch.no_grad():
            generated_image = generator(input_image)
        
        # Post-traitement de l'image générée
        generated_image = generated_image[0].detach().cpu()
        generated_image = (generated_image * 0.5 + 0.5)  # Dénormalisation

        # Sauvegarder l'image générée
        output_path = os.path.join(output_folder, f"generated_{filename}")
        #generated_image.save(output_path)
        generated_image_np = (generated_image.squeeze().numpy() * 255).astype('uint8')
        generated_image_np = generated_image_np - generated_image_np.min()
        generated_image_np = generated_image_np / generated_image_np.max() * 255
        generated_image_np = generated_image_np.astype('uint8')
        generated_image_pil = Image.fromarray(generated_image_np, mode='L')
        generated_image_pil.save(output_path) 
        print("Forme de l'image générée :", generated_image_np.shape)
        print("Type de l'image générée :", generated_image_np.dtype)
        print("Valeurs min/max :", generated_image_np.min(), generated_image_np.max())
        print(f"Image générée sauvegardée : {output_path}")

print("Toutes les images ont été générées et sauvegardées.")
