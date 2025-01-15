import os
import shutil
import random
from pathlib import Path
import math

def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    print(f"train_ratio: {train_ratio}, val_ratio: {val_ratio}, test_ratio: {test_ratio}")
    print(f"Sum of ratios: {train_ratio + val_ratio + test_ratio}")
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-9), "Sum of ratios must be 1.0"

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(folder, exist_ok=True)

    # Rekurencyjne przeszukiwanie katalogów
    images = list(Path(dataset_dir).rglob("*.jpg")) + list(Path(dataset_dir).rglob("*.png"))
    annotations = list(Path(dataset_dir).rglob("*.txt"))

    print(f"Znalezione obrazy: {len(images)}")
    print(f"Znalezione adnotacje: {len(annotations)}")

    # Dopasowanie obrazów i adnotacji
    paired_data = []
    for img_path in images:
        annotation_path = img_path.with_suffix(".txt")
        if annotation_path.exists():
            paired_data.append((img_path, annotation_path))
        else:
            print(f"Brak adnotacji dla obrazu: {img_path}")

    print(f"Dopasowane pary: {len(paired_data)}")
    random.shuffle(paired_data)

    num_total = len(paired_data)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)

    train_data = paired_data[:num_train]
    val_data = paired_data[num_train:num_train + num_val]
    test_data = paired_data[num_train + num_val:]

    for data, folder in [(train_data, train_dir), (val_data, val_dir), (test_data, test_dir)]:
        for image_path, label_path in data:
            print(f"Kopiowanie: {image_path} -> {folder}")
            print(f"Kopiowanie: {label_path} -> {folder}")
            shutil.copy(image_path, folder)
            shutil.copy(label_path, folder)

    print(
        f"Dataset podzielony: {len(train_data)} treningowych, {len(val_data)} walidacyjnych, {len(test_data)} testowych.")

# Przykład użycia
dataset_dir = r"C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\cars_make_models_reduced"
output_dir = r"C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\divided_set"
split_dataset(dataset_dir, output_dir)
