import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm  # Importation de tqdm

def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Met le modèle en mode entraînement
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Création de la barre de progression avec tqdm pour l'entraînement
    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Mise à jour de la barre de progression avec la perte et la précision
            pbar.set_postfix(loss=running_loss / (pbar.n + 1), accuracy=100 * correct / total)
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()
    valid_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = valid_loss / len(data_loader)
    return avg_loss, accuracy