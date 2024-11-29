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

checkpoint_path = os.path.join('/home/ychappetjuan/free-projetIA12456/logs/run_20241122_140602/checkpoints', 'checkpoint_25.pth')
checkpoint = torch.load(checkpoint_path, map_location=device)

if 'generator_state_dict' in checkpoint:
    generator.load_state_dict(checkpoint['generator_state_dict'])
else:
    print(f"Erreur: 'generator_state_dict' n'est pas trouvé dans {checkpoint_path}")
    exit()

generator.eval()

image_path = '/home/ychappetjuan/free-projetIA12456/MASK/mask_transform/000_CNV_1_mask_transform.jpg'
image = Image.open(image_path).convert('RGB')


generator.load_state_dict(torch.load(os.path.join('/Users/yannischappetjuan/Desktop/IA/PROJET/free-projetIA12456/runs/run_20241122_103458/checkpoints/', 'best_model.pth')))
generator.eval()

image = Image.open('/Users/yannischappetjuan/Desktop/IA/PROJET/free-projetIA12456/MASK/mask_transform/000_CNV_3_mask_transform.jpg').convert('RGB')

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  
])

input_image = transform(image).unsqueeze(0).to(device)  

with torch.no_grad():
    generated_image = generator(input_image)

generated_image = generated_image[0].detach().cpu() 
generated_image = (generated_image * 0.5 + 0.5)  
generated_image = generated_image.permute(1, 2, 0)  
generated_image = (generated_image * 255).byte()  

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.subplot(1, 2, 2)
plt.imshow(generated_image)
plt.title("Generated Image")
plt.show()
