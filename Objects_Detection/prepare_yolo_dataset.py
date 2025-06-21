import os
import shutil
import random
from pathlib import Path

SOURCE_DATA_BASE_PATH = "dataset_2d_shapes"
YOLO_DATASET_BASE_PATH = "yolo_shapes_dataset"
TRAIN_SPLIT_RATIO = 0.8

SHAPE_NAMES = ["Rectangle", "Square", "Circle", "Triangle"]
for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_DATASET_BASE_PATH, "images", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_BASE_PATH, "labels", split), exist_ok=True)

print(f"Created directory structure under: {YOLO_DATASET_BASE_PATH}")

for shape_name in SHAPE_NAMES:
    print(f"\nProcessing shape: {shape_name}")

    source_image_dir = Path(SOURCE_DATA_BASE_PATH) / "images" / shape_name
    source_label_dir = Path(SOURCE_DATA_BASE_PATH) / "labels" / shape_name

    image_files = sorted([f for f in source_image_dir.glob("*.png")])
    
    if not image_files:
        print(f"  No image files found for {shape_name} in {source_image_dir}")
        continue

    random.shuffle(image_files)

    num_train = int(len(image_files) * TRAIN_SPLIT_RATIO)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    print(f"  Total images: {len(image_files)}")
    print(f"  Training images: {len(train_files)}")
    print(f"  Validation images: {len(val_files)}")

    def copy_files(file_list, split_type):
        for img_file_path in file_list:
            label_file_name = img_file_path.stem + ".txt"
            label_file_path = source_label_dir / label_file_name

            dest_img_path = Path(YOLO_DATASET_BASE_PATH) / "images" / split_type / img_file_path.name
            dest_label_path = Path(YOLO_DATASET_BASE_PATH) / "labels" / split_type / label_file_name
            
            if img_file_path.exists():
                shutil.copy(img_file_path, dest_img_path)
            else:
                print(f"    Warning: Image file not found: {img_file_path}")

            if label_file_path.exists():
                shutil.copy(label_file_path, dest_label_path)
            else:
                print(f"    Warning: Label file not found for {img_file_path.name} at {label_file_path}")

    print(f"  Copying training files for {shape_name}...")
    copy_files(train_files, "train")
    
    print(f"  Copying validation files for {shape_name}...")
    copy_files(val_files, "val")

print("\nDataset preparation complete!")
print(f"YOLOv8 compatible dataset is ready in: {YOLO_DATASET_BASE_PATH}")