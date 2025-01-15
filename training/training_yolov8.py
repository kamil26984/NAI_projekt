import os
import torch
from ultralytics import YOLO

# Ścieżki do zbioru danych i konfiguracji
# data_path = r"C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\divided_set"  # Ścieżka do podzielonego datasetu
model_output = "runs/train"  # Katalog na zapisany model
data_yaml = r"C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\divided_set\data.yaml"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# if device == 'cuda':
#     print("GPU jest dostępne i będzie używane do treningu.")
# else:
#     print("GPU nie jest dostępne, używane będzie CPU.")

# Tworzenie folderów, jeśli nie istnieją
os.makedirs(model_output, exist_ok=True)

# # Tworzenie i trenowanie modelu
# model = YOLO("yolov8n.pt")  # Wybór architektury (np. yolov8n, yolov8s)
# model.train(
#     data="C:/Users/Admin/Desktop/NAI PROEJKT/NAI_projekt/divided_set/data.yaml",  # Plik YAML opisujący dane
#     epochs=50,                     # Liczba epok
#     imgsz=640,                     # Rozmiar obrazu
#     batch=16,                      # Rozmiar batcha
#     name="car_brand_recognition",  # Nazwa eksperymentu
#     save_period=5,                 # Zapis co 5 epok
#     device=device                 # Ustawienie urządzenia (GPU/CPU)
# )

if __name__ == "__main__":  # To jest kluczowe dla Windows
    # Kod, który uruchamia trening
    model = YOLO("yolov8n.pt")
    # model.val(data=data_yaml)
    model.train(
        data=data_yaml,  # Plik YAML opisujący dane
        epochs=50,  # Liczba epok
        imgsz=640,  # Rozmiar obrazu
        batch=8,  # Rozmiar batcha
        name="car_brand_recognition",  # Nazwa eksperymentu
        save_period=5,  # Zapis co 5 epok
        device=device  # Ustawienie urządzenia (GPU/CPU)
    )






# import os
# from ultralytics import YOLO
#
# # Ścieżki do zbioru danych i konfiguracji
# data_yaml = r"C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\divided_set\data.yaml"
# model_output = r"runs/train"  # Katalog na zapisany model
#
# # Sprawdzanie, czy GPU jest dostępne
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # print(f"Urządzenie używane do treningu: {device}")
#
# # Tworzenie folderów, jeśli nie istnieją
# os.makedirs(model_output, exist_ok=True)
#
# if __name__ == "__main__":
#     # Tworzenie i trenowanie modelu
#     model = YOLO("yolov8n.pt")  # Wybór pretrenowanego modelu YOLOv8n
#     model.train(
#         data=data_yaml,  # Plik YAML opisujący dane
#         epochs=50,       # Liczba epok
#         imgsz=640,       # Rozmiar obrazu
#         batch=8,         # Rozmiar batcha
#         name="car_brand_recognition",  # Nazwa eksperymentu
#         save_period=5,   # Zapis co 5 epok
#         device='cpu'    # Ustawienie urządzenia (GPU/CPU)
#     )
