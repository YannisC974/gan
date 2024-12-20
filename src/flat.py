import os
import shutil
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
source_dir = os.getenv('SOURCE_DIR')

def flatten_images(input_dir, output_dir):
    """
    Flatten all images from a directory and its subdirectories into a single directory.
    Handles duplicate filenames by appending a unique suffix.
    Converts all images to .jpg format.

    Args:
        input_dir (str): Root directory containing subdirectories with images.
        output_dir (str): Directory where all flattened images will be stored.
    """
    os.makedirs(output_dir, exist_ok=True) 
    processed_files = set()  

    for root, _, files in os.walk(input_dir):
        for filename in files:
        
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
                input_path = os.path.join(root, filename)

                # Ensure unique filenames in the output directory with .jpg extension
                base_name, _ = os.path.splitext(filename)
                counter = 1
                new_name = f"{base_name}.jpg"
                output_path = os.path.join(output_dir, new_name)
                while output_path in processed_files:
                    new_name = f"{base_name}_{counter}.jpg"
                    output_path = os.path.join(output_dir, new_name)
                    counter += 1

                # Convert to .jpg if not already in .jpg format
                try:
                    if not filename.lower().endswith('.jpg'):
                        with Image.open(input_path) as img:
                            rgb_img = img.convert('RGB')  # Convert to RGB (required for JPEG)
                            rgb_img.save(output_path, 'JPEG')
                    else:
                        shutil.copy(input_path, output_path)  # Copy .jpg files directly
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")
                    continue

                processed_files.add(output_path)

    print(f"All images have been flattened into '{output_dir}'.")

# Usage
input_directory = os.path.join(source_dir, 'dataset')
output_directory = os.path.join(source_dir, 'dataset_flat')
flatten_images(input_directory, output_directory)
