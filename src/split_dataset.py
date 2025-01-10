import os
from torchvision import transforms
from dataset import EyeDataset, split_dataset_balanced, save_split_to_folders
from dotenv import load_dotenv

load_dotenv()
root_dir = os.getenv("ROOT_DIR")

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = EyeDataset(root_dir=root_dir, transform=transform)
    subset1, subset2 = split_dataset_balanced(dataset)
    save_split_to_folders(subset1, subset2, output_dir="output")

if __name__ == "__main__":
    main()