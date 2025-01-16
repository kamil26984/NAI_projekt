import os
from ultralytics import YOLO


def train_yolo(dataset_path, model_name="yolo11s.pt", epochs=40, batch_size=32, img_size=640, patience=10):
    """
    Trenuj model YOLO na dostarczonym zestawie danych z lekką augmentacją.

    """
    model = YOLO(model_name)

    # Rozpocznij trening
    model.train(
        data=dataset_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        optimizer="AdamW",
        patience=patience,  # Early stopping
        workers=4,
        device=0,  # GPU
        project="YOLO_Training",
        name="car_detection"
    )

    print("Trening zakończony")


if __name__ == "__main__":
    dataset_path = "data.yaml"
    model_name = "yolo11s.pt"
    train_yolo(dataset_path, model_name=model_name, epochs=50, batch_size=16, img_size=640, patience=10)
