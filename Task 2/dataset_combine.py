# combine_datasets.py

import os
import shutil

def merge_folders(train_dir, val_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for source_dir in [train_dir, val_dir]:
        for class_name in os.listdir(source_dir):
            src_class_path = os.path.join(source_dir, class_name)
            dst_class_path = os.path.join(output_dir, class_name)

            if not os.path.isdir(src_class_path):
                continue

            os.makedirs(dst_class_path, exist_ok=True)

            for file in os.listdir(src_class_path):
                src_file = os.path.join(src_class_path, file)
                dst_file = os.path.join(dst_class_path, file)

                # Avoid overwriting files with same name
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dst_file):
                        dst_file = os.path.join(dst_class_path, f"{base}_{counter}{ext}")
                        counter += 1

                shutil.copy2(src_file, dst_file)

    print(f"[✓] Merged dataset saved at: {output_dir}")

if __name__ == "__main__":
    merge_folders(
        train_dir='data/Comsys/Comsys_Hackathon5/Task_B/train',
        val_dir='data/Comsys/Comsys_Hackathon5/Task_B/val',
        output_dir='dataset_combined'
    )
