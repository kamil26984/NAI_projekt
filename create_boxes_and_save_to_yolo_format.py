import os
from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
print(torch.cuda.is_available())  # Powinno zwrócić True, jeśli GPU jest dostępne
print(torch.cuda.current_device())  # Numer urządzenia GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Nazwa GPU


# Mapa marek do klas (przykład, możesz rozszerzyć lub zmienić)
brand_to_class_map = {
    "audi": 0,
    "bmw": 1,
    "chevrolet": 2,
    "dodge": 3,
    "ford": 4,
    "honda": 5,
    "hyundai": 6,
    "jeep": 7,
    "kia": 8,
    "landrover": 9,
    "lexus": 10,
    "mazda": 11,
    "mercedes": 12,
    "nissan": 13,
    "peugeot": 14,
    "porsche": 15,
    "toyota": 16,
}

# Sprawdzanie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Inicjalizacja modelu i procesora
processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50", use_fast=True)
model = ConditionalDetrForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")

# Przeniesienie modelu na GPU (jeśli dostępne)
model.to(device)
print(model.config.id2label)

# Funkcja wizualizacji
def visualize_detections(image, results, output_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        box = box.tolist()
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            box[0], box[1] - 10,
            f"{model.config.id2label[label.item()]}: {score:.2f}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.7)
        )
    plt.axis("off")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# Funkcja generowania anotacji
def generate_annotations(csv_file, vis_dir=None):
    with open(csv_file, 'r') as file:
        lines = file.readlines()[1:]  # Pomijamy nagłówek

    if vis_dir and not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    for line in tqdm(lines, desc="Processing images"):
        image_path, brand = line.strip().split(",")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        if brand not in brand_to_class_map:
            print(f"Unknown brand: {brand}, skipping.")
            continue

        # Pobierz numer klasy z mapy
        class_id = brand_to_class_map[brand]

        # Ładowanie obrazu
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        # Przeniesienie danych na GPU (jeśli dostępne)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Wykrywanie obiektów
        with torch.no_grad():
            outputs = model(**inputs)

        # Przetwarzanie wyników
        target_sizes = torch.tensor([image.size[::-1]]).to(device)  # (height, width)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        # Filtrowanie i logowanie wyników detekcji
        if results["scores"].shape[0] == 0:
            print(f"No detections for {image_path}")
            continue

        # Wybór największego boxa na podstawie powierzchni
        max_area = 0
        best_box = None
        best_score = None
        for score, box in zip(results["scores"], results["boxes"]):
            box_list = box.tolist()
            width = box_list[2] - box_list[0]
            height = box_list[3] - box_list[1]
            area = width * height
            if area > max_area:
                max_area = area
                best_box = box
                best_score = score

        if best_box is None:
            print(f"No valid detections for {image_path}")
            continue

        # Ścieżka do pliku .txt
        annotation_path = os.path.splitext(image_path)[0] + ".txt"
        with open(annotation_path, 'w') as f:
            box = best_box.tolist()

            # Normalizacja do formatu YOLO
            x_center = ((box[0] + box[2]) / 2) / image.size[0]
            y_center = ((box[1] + box[3]) / 2) / image.size[1]
            width = (box[2] - box[0]) / image.size[0]
            height = (box[3] - box[1]) / image.size[1]

            print(f"Image: {image_path}, Brand: {brand}, Mapped Class ID: {class_id}, Score: {best_score:.2f}")
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {best_score:.6f}\n")

        # Zapis wizualizacji (opcjonalnie)
        if vis_dir:
            vis_path = os.path.join(vis_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
            visualize_detections(image, {"boxes": [best_box], "scores": [best_score], "labels": [torch.tensor(class_id)]}, output_path=vis_path)

    print(f"Annotations saved as .txt files next to corresponding images.")

if __name__ == "__main__":
    csv_file = "cars_make_models_reduced/annotations.csv"
    vis_dir = "cars_make_models_reduced/detr_visualizations"  # Folder na wizualizacje (opcjonalne)
    generate_annotations(csv_file)
