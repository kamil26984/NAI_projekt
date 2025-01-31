import gradio as gr
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import numpy as np

base_path = os.path.dirname(__file__)

car_detector = YOLO(os.path.join(base_path, "yolov5su.pt"))  # Model do wykrywania samochodów ale same obiekty bez marek
brand_detector = {
    "YOLOv8": YOLO(os.path.join(base_path, "YOLOv8m_car_detection", "weights", "best.pt")),  # ścieżki do wytrenowanych modeli
    "YOLOv9": YOLO(os.path.join(base_path, "YOLOv9t_car_detection", "weights", "best.pt")),
    "YOLOv11": YOLO(os.path.join(base_path, "YOLOv11s_car_detection", "weights", "best.pt")),
    "YOLOv5": YOLO(os.path.join(base_path, "YOLOv5n_car_detection", "weights", "best.pt"))
}

def detect_cars(image):
    #wykrywanie samych samochodow ze zdjecia
    results = car_detector.predict(source=image, conf=0.6, classes=(2,5,7))
    return results[0].boxes.xyxy  # Bounding boxy


def classify_brand(image, models):
    #wykrywanie marki samochodu ze zdjecia z wycietym samochodem
    output_images = {}


    img_array = np.array(image)[:, :, ::-1]  # rgb na bgr

    for model_name, model in models.items():
        result = model.predict(source=img_array)
        output_image = result[0].plot()
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB) #zamiana z powrotem na rgb
        output_images[model_name] = Image.fromarray(output_image_rgb)

    return output_images


def process_image(image_path):
    #przetwarzanie wycietych zdjec + rozpoznawanie marki
    image = Image.open(image_path)
    boxes = detect_cars(image_path)

    results = {"YOLOv8": [], "YOLOv9": [], "YOLOv11": [], "YOLOv5": []}

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_car = image.crop((x_min, y_min, x_max, y_max))

        classified_images = classify_brand(cropped_car, brand_detector)


        for model_name, output_image in classified_images.items():
            results[model_name].append(output_image)

    return results["YOLOv8"], results["YOLOv9"], results["YOLOv11"], results["YOLOv5"]


interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Gallery(type="pil", label="YOLOv8"),
        gr.Gallery(type="pil", label="YOLOv9"),
        gr.Gallery(type="pil", label="YOLOv11"),
        gr.Gallery(type="pil", label="YOLOv5")
    ],
    title="Car make recognition",
    description="Upload a car photo. Car makes that should be detected: Audi, BMW, Chevrolet, Dodge, Ford, Honda, Hyundai, Jeep, Kia, Landrover, Lexus, Mazda, Mercedes, Nissan, Peugeot, Porsche and Toyota"
)

if __name__ == "__main__":
    interface.launch()
