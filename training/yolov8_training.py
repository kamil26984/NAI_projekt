import os
import torch
from ultralytics import YOLO


# Funkcja do trenowania modelu YOLO
def train_yolo(dataset_path, model_name="yolo11s.pt", epochs=50, batch_size=8, img_size=640, patience=10):
    """
    Trenuj model YOLO na dostarczonym zestawie danych z lekką augmentacją.

    """
    model_output = "runs/train"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Załaduj model YOLO
    model = YOLO(model_name)
    os.makedirs(model_output, exist_ok=True)

    # Rozpocznij trening
    model.train(
        data=dataset_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        optimizer="AdamW",
        patience=patience,  # Early stopping
        workers=4,
        device=device,  # GPU
        project="YOLO_Training",
        name="car_detection"
    )

    print("Trening zakończony")


# Test funkcji
if __name__ == "__main__":
    # Ścieżka do folderu z danymi
    dataset_path = r"C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\divided_set\data.yaml"
    # Model YOLOv8 do wykorzystania
    model_name = "yolov8m.pt"  # Możesz zmienić na yolov8s.pt lub większy
    # Rozpocznij trening
    #40
    train_yolo(dataset_path, model_name=model_name, epochs=50, batch_size=8, img_size=640, patience=10)
