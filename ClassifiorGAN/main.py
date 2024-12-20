import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import TernaryClassifier
from dataset import EyeDataset, split_dataset_balanced
from engine import train, evaluate, count_class_inferences
from model import TernaryClassifier
from engine import train, evaluate
from collections import Counter
import os
from dotenv import load_dotenv
from torch.utils.data import Subset, random_split

load_dotenv()
root_dir = os.getenv("ROOT_DIR")

def main():
    # Configuration du périphérique (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Chargement des données avec les transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = EyeDataset(root_dir=root_dir, transform=transform)
    subset1, subset2 = split_dataset_balanced(train_dataset)
    
    # def check_distribution(dataset):
    #     labels = [train_dataset[i][1] for i in range(len(dataset))]
    #     return Counter(labels)

    # print("Distribution dans subset1:", check_distribution(subset1))
    # print("Distribution dans subset2:", check_distribution(subset2))

    train_ratio = 0.8  # 80% pour l'entraînement, 20% pour la validation
    train_size = int(train_ratio * len(subset1))
    valid_size = len(subset1) - train_size

    # Séparer subset1 en train et valid
    train_data, valid_data = random_split(subset1, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

    # # Initialisation du modèle
    model = TernaryClassifier().to(device)
    
    # # Définition de la fonction de perte et de l'optimiseur
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # # Entraînement du modèle
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    
    # Évaluation du modèle
    test_accuracy = evaluate(model, valid_loader, device)  # Vous pouvez utiliser un DataLoader de test ici
    print(f"Validation Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
