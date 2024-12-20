import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from tqdm import tqdm  # Pour la barre de progression
import matplotlib.pyplot as plt

# Définir le modèle
class TernaryClassifier(nn.Module):
    def __init__(self):
        super(TernaryClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),  # Adapter la taille en fonction des dimensions de l'image
            nn.ReLU(),
            nn.Linear(128, 3),  # Sortie pour 3 classes
            nn.Softmax(dim=1)  # Activation Softmax pour 3 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



