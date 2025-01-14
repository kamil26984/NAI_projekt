import os
import shutil
import random
from pathlib import Path
import math


# Funkcja do podziału datasetu na treningowy, walidacyjny i testowy
# nie w tej postaci, ale część logiki przyda nam się do podziału tego, nad którym pracujemy



def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Dzieli dataset na treningowy, walidacyjny i testowy.
    """
    # Sprawdzamy, czy suma proporcji wynosi 1.0
    print(f"train_ratio: {train_ratio}, val_ratio: {val_ratio}, test_ratio: {test_ratio}")
    print(f"Sum of ratios: {train_ratio + val_ratio + test_ratio}")
    #assert train_ratio + val_ratio + test_ratio == 1.0, "Sum of ratios must be 1.0"
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-9), "Sum of ratios must be 1.0" # komputer sam nie umie liczyć, więc trzeba użyć math.isclose

    # Ścieżki wyjściowe
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(folder, exist_ok=True)

    # Pobieramy wszystkie pliki w katalogu
    #images = list(Path(dataset_dir).glob("*.jpg"))  # Zakładamy, że obrazy są w formacie JPG a jeśli nie są to zmień
    images = list(Path(dataset_dir).glob("*.mp4")) + list(Path(dataset_dir).glob("*.mov"))
    annotations = list(Path(dataset_dir).glob("*.txt"))  # Pliki etykiet

    # Upewniamy się, że dla każdego obrazu jest odpowiedni plik etykiet
    images.sort()
    annotations.sort()
    #paired_data = list(zip(images, annotations))
    paired_data = [(images, annotations) for images, annotations in zip(images, annotations) if images.stem == annotations.stem]
    random.shuffle(paired_data)  # Mieszamy dane

    # Podział danych
    num_total = len(paired_data)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)

    train_data = paired_data[:num_train]
    val_data = paired_data[num_train:num_train + num_val]
    test_data = paired_data[num_train + num_val:]

    # Przenoszenie plików
    for data, folder in [(train_data, train_dir), (val_data, val_dir), (test_data, test_dir)]:
        for image_path, label_path in data:
            shutil.copy(image_path, folder)
            shutil.copy(label_path, folder)

    print(
        f"Dataset podzielony: {len(train_data)} treningowych, {len(val_data)} walidacyjnych, {len(test_data)} testowych.")


# Przykład użycia
dataset_dir = "datasets/vriv"  # Ścieżka do aktualnego datasetu
output_dir = "datasets/vriv_divided"  # Katalog na podzielony dataset
split_dataset(dataset_dir, output_dir)
