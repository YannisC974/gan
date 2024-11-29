import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def visualize_batch_images(real_images, fake_images, masks):
    """
    Visualise un batch d'images réelles, générées et masques
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Real Images')
    plt.imshow(vutils.make_grid(real_images.cpu(), normalize=True).permute(1, 2, 0))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Generated Images')
    plt.imshow(vutils.make_grid(fake_images.cpu(), normalize=True).permute(1, 2, 0))
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Masks')
    plt.imshow(vutils.make_grid(masks.repeat(1, 3, 1, 1), normalize=True).permute(1, 2, 0))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_one_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, device, bool, gradient_clip_value=1.0):
    generator.train()
    discriminator.train()

    total_loss_D = 0
    total_loss_G = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc='Training', leave=False)

    for batch_idx, (real_images, masks) in enumerate(progress_bar):
        batch_size = real_images.size(0)

        if real_images.size(1) != 3:
            raise ValueError(f"Expected 3 channels for real images, got {real_images.size(1)}")

        real_images = real_images.to(device)
        masks = masks.to(device)

        if masks.size(1) == 1:
            masks_expanded = masks.repeat(1, 3, 1, 1)
        else:
            masks_expanded = masks
        try:
            optimizer_D.zero_grad()

            with torch.no_grad():
                fake_images = generator(masks_expanded)

            real_input = torch.cat((real_images, masks), dim=1)
            fake_input = torch.cat((fake_images.detach(), masks), dim=1)

            output_real = discriminator(real_input)
            output_fake = discriminator(fake_input)

            label_real = torch.ones_like(output_real, device=device)
            label_fake = torch.zeros_like(output_fake, device=device)

            loss_D_real = criterion(output_real, label_real)
            loss_D_fake = criterion(output_fake, label_fake)
            loss_D = loss_D_real + loss_D_fake

            loss_D.backward()
            optimizer_D.step()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), gradient_clip_value)

            optimizer_G.zero_grad()

            fake_images = generator(masks_expanded)

            # if batch_idx % 10 == 0:  # Visualise tous les 10 batches
            #     visualize_batch_images(real_images, fake_images.cpu(), masks.cpu())

            output_fake_for_G = discriminator(torch.cat((fake_images, masks), dim=1))

            loss_G = criterion(output_fake_for_G, label_real)

            loss_G.backward()
            optimizer_G.step()

            torch.nn.utils.clip_grad_norm_(generator.parameters(), gradient_clip_value)

            total_loss_D += loss_D.item() 
            total_loss_G += loss_G.item()

            progress_bar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}'
            })

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            print(f"Shapes - real_images: {real_images.shape}, masks: {masks.shape}, fake_images: {fake_images.shape}")
            print(f"Devices - real_images: {real_images.device}, masks: {masks.device}, fake_images: {fake_images.device}")
            continue

    avg_loss_D = total_loss_D / num_batches
    avg_loss_G = total_loss_G / num_batches

    return avg_loss_D, avg_loss_G


def validate(generator, discriminator, dataloader, criterion, device):
    generator.eval()
    discriminator.eval()

    total_loss_D = 0
    total_loss_G = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for real_images, masks in dataloader:
            # Expand masks if needed
            if masks.size(1) == 1:
                masks_expanded = masks.repeat(1, 3, 1, 1)
            else:
                masks_expanded = masks

            real_images = real_images.to(device)
            masks_expanded = masks_expanded.to(device)
            masks = masks.to(device) 

            fake_images = generator(masks_expanded)

            real_input = torch.cat((real_images, masks), dim=1)  
            fake_input = torch.cat((fake_images, masks), dim=1)

            output_real = discriminator(real_input)
            output_fake = discriminator(fake_input)

            label_real = torch.ones_like(output_real).to(device)
            label_fake = torch.zeros_like(output_fake).to(device)

            loss_D = criterion(output_real, label_real) + criterion(output_fake, label_fake)
            loss_G = criterion(output_fake, label_real)

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()

    return total_loss_D / num_batches, total_loss_G / num_batches
