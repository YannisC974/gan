import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from engine import train_one_epoch
from datetime import datetime

from dataset import Dataset
from GAN import UNet, Discriminator
from utils import EarlyStopping

torch.cuda.empty_cache()

def weights_init(m):
    """Initialize network weights."""
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(state, filename):
    """Save checkpoint of the model."""
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def main(args):
    # Set up device
    device = torch.device('mps' if torch.backends.mps.is_available() 
                         else 'cuda' if torch.cuda.is_available() 
                         else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.log_dir, f'run_{timestamp}')
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    sample_dir = os.path.join(run_dir, 'samples')
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=run_dir)

    # Load dataset
    dataset_images = Dataset(root_dir=args.image_dir)
    dataset_masks = Dataset(root_dir=args.mask_dir)

    # Split dataset
    # train_size = int(0.8 * len(dataset_images))
    # val_size = len(dataset_images) - train_size
    # train_dataset, val_dataset = random_split(
    #     dataset_images, 
    #     [train_size, val_size],
    #     generator=torch.Generator().manual_seed(42)
    # )

    # Create data loaders
    train_loader = DataLoader(
        dataset=dataset_images,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )

    mask_loader = DataLoader(
        dataset=dataset_masks,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize models
    generator = UNet().to(device)
    discriminator = Discriminator().to(device)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Optimizers
    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=0.00002,
        betas=(args.beta1, 0.999)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=0.000005,
        betas=(args.beta1, 0.999)
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=os.path.join(checkpoint_dir, 'best_model.pth')
    )

    # Training loop
    for epoch in range(args.epochs):
        # Training
        print(epoch)
        train_loss_D, train_loss_G = train_one_epoch(
            generator, 
            discriminator, 
            train_loader, 
            mask_loader, 
            optimizer_G, 
            optimizer_D, 
            criterion, 
            device, 
            epoch,
            writer,
        )

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train_D', train_loss_D, epoch)
        writer.add_scalar('Loss/Train_G', train_loss_G, epoch)

        # Check for early stopping
        early_stopping(train_loss_G, generator)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAN for Image Inpainting')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the images directory')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to the masks directory')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.000001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval to save model')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')

    args = parser.parse_args()
    main(args)
