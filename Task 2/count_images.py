# count_images_folders.py

import os

def count_folders_and_images(dataset_dir):
    folder_count = 0
    image_count = 0

    for root, dirs, files in os.walk(dataset_dir):
        # Count only top-level class folders
        if root == dataset_dir:
            folder_count += len(dirs)
        else:
            image_count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"[✓] Total Class Folders  : {folder_count}")
    print(f"[✓] Total Image Files   : {image_count}")

if __name__ == "__main__":
    dataset_path = 'dataset_combined'  # change if needed
    count_folders_and_images(dataset_path)
