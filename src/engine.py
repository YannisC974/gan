import torch 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision

def train_one_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, device, epoch, writer, gradient_clip_value=1.0):
    generator.train()
    discriminator.train()

    total_loss_D = 0
    total_loss_G = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    l1_loss = torch.nn.L1Loss()

    for batch_idx, (real_images, masks) in enumerate(progress_bar):
        batch_size = real_images.size(0)

        if real_images.size(1) != 1:
            raise ValueError(f"Expected 1 channels for real images, got {real_images.size(1)}")

        real_images = real_images.to(device)
        masks = masks.to(device)

        try:
            optimizer_D.zero_grad()

            with torch.no_grad():
                fake_images = generator(masks)

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

            fake_images = generator(masks)
            output_fake_for_G = discriminator(torch.cat((fake_images, masks), dim=1))

            loss_G_C = criterion(output_fake_for_G, label_real)
            loss_G_l1 = l1_loss(fake_images, real_images)
            loss_G = loss_G_C + 100*loss_G_l1

            loss_G.backward()
            optimizer_G.step()

            torch.nn.utils.clip_grad_norm_(generator.parameters(), gradient_clip_value)

            total_loss_D += loss_D.item() 
            total_loss_G += loss_G.item()

            progress_bar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}'
            })

            if batch_idx % 10 == 0:
                img_grid_real = torchvision.utils.make_grid(real_images[:8])
                img_grid_fake = torchvision.utils.make_grid(fake_images[:8])

            writer.add_image("Fake", img_grid_fake, global_step=(epoch * num_batches) + batch_idx)
            writer.add_image("Real", img_grid_real, global_step=(epoch * num_batches) + batch_idx)

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            print(f"Shapes - real_images: {real_images.shape}, masks: {masks.shape}, fake_images: {fake_images.shape}")
            print(f"Devices - real_images: {real_images.device}, masks: {masks.device}, fake_images: {fake_images.device}")
            continue

    avg_loss_D = total_loss_D / num_batches
    avg_loss_G = total_loss_G / num_batches

    return avg_loss_D, avg_loss_G

