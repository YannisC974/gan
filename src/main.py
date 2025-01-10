import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from dotenv import load_dotenv
from collections import Counter

from model import TernaryClassifier
from dataset import EyeDatasetOutput
from engine import train, evaluate

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Charger les variables d'environnement
load_dotenv()
classifior_dir = os.getenv("CLASSIFIOR_DIR")

def plot_confusion_matrix(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["CNV", "DRUSEN", "NORMAL"], yticklabels=["CNV", "DRUSEN", "NORMAL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def main():
    # Configuration du périphérique (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Chargement des données avec les transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    classifior_dataset = EyeDatasetOutput(root_dir=classifior_dir, transform=transform)

    # Division des données en ensemble d'entraînement et de validation
    train_ratio = 0.8  # 80% pour l'entraînement, 20% pour la validation
    train_size = int(train_ratio * len(classifior_dataset))
    valid_size = len(classifior_dataset) - train_size
    train_data, valid_data = random_split(classifior_dataset, [train_size, valid_size])

    train_labels = [label for _, label in train_data]
    valid_labels = [label for _, label in valid_data]
    print("Train class distribution:", Counter(train_labels))
    print("Validation class distribution:", Counter(valid_labels))

    # Création des DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

    # Initialisation du modèle
    model = TernaryClassifier().to(device)

    # Définition de la fonction de perte et de l'optimiseur
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping
    early_stopping_patience = 5
    best_valid_loss = float('inf')
    patience_counter = 0

    # Entraînement du modèle
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # Phase d'entraînement
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Phase de validation
        valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion, device)
        print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")

        # Early stopping logic
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            # Sauvegarde du meilleur modèle
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    plot_confusion_matrix(model, valid_loader, device)
    print("Entraînement terminé.")

if __name__ == "__main__":
    main()