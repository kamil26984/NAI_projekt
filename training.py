from ultralytics import YOLO
import torchvision.transforms as T
import os
import random

STANFORD_CARS_IMG = "datasets/yolo_format/train"
STANFORD_CARS_DATA = "datasets/yolo_format/data.yaml"
STANFORD_CARS_LABELS = "datasets/yolo_format/train"
CARS_DATASET_IMG = "datasets/cars_dataset/train/images"
CARS_DATASET_DATA = "datasets/cars_dataset/data.yaml"
CARS_DATASET_LABELS = "datasets/cars_dataset/train/labels"

CARS_DATASET = "datasets/cars_dataset/train"

# Transformacje danych
data_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),  # Odbicie poziome
    T.RandomRotation(degrees=15),  # Rotacja
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Zmiany kolorów
    T.ToTensor(),  # Konwersja do tensora
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizacja
])


def train_model(subset_size=None):
    # Funkcja wybierająca ograniczony subset zdjęć
    def get_subset(images_dir, labels_dir, subset_size):
        all_images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
        all_labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.txt')])

        if subset_size is not None and subset_size < len(all_images):
            # Losowe wybranie próbek
            subset_indices = random.sample(range(len(all_images)), subset_size)
            subset_images = [all_images[i] for i in subset_indices]
            subset_labels = [all_labels[i] for i in subset_indices]
        else:
            subset_images = all_images
            subset_labels = all_labels

        return subset_images, subset_labels

    #images, labels = get_subset(STANFORD_CARS_IMG, STANFORD_CARS_LABELS, subset_size)

    """
    # Tworzenie datasetu
    dataset = CustomDataset(
        images=images,
        labels=labels,
        transform=data_transforms
    )"""

    model = YOLO("yolo11s.pt")

    model.train(
        data=STANFORD_CARS_DATA,
        epochs=3,
        batch=16,
        imgsz=640,
        optimizer="Adam",
        lr0=1e-4
    )

    metrics = model.val()
    print("Wyniki walidacji:", metrics)

    model.export(format="onnx", dynamic=True, simplify=False)


if __name__ == '__main__':
    train_model(subset_size=500)
