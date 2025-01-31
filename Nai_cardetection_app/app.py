from ultralytics import YOLO
import gradio as gr
import cv2
import torch

from PIL import Image

#TA APKA WYKRYWA TYLKO JAK JEST JEDEN SAMOCHOD NA ZDJECIU


modelv11s = YOLO (r"C:\Users\mszym\nai\Nai_cardetection_app\YOLOv9t_car_detection\weights\best.pt")
modelv9t = YOLO (r"C:\Users\mszym\nai\Nai_cardetection_app\YOLOv11s_car_detection\weights\best.pt")
modelv8m = YOLO (r"C:\Users\mszym\nai\Nai_cardetection_app\YOLOv8m_car_detection\weights\best.pt")

models = [modelv11s, modelv9t, modelv8m]

def predict(image):
    # predictions = []
    # model yolov9
    yolov9_results = modelv9t(image)
    output_image_model9 = yolov9_results[0].plot()
    output_image_rgb_model9 = cv2.cvtColor(output_image_model9, cv2.COLOR_BGR2RGB)
    yolo9_prediction = Image.fromarray(output_image_rgb_model9)
    #model yolov11
    yolov11_results = modelv11s(image)
    output_image_model11 = yolov11_results[0].plot()
    output_image_rgb_model11 = cv2.cvtColor(output_image_model11, cv2.COLOR_BGR2RGB) #zmiana koloru na rgb bo plot() zmienia barwy
    yolo11_prediction = Image.fromarray(output_image_rgb_model11)

    # for model in models:
    #     result = model(image)
    #     output_image_model = result[0].plot()
    #     output_image_rgb_model = cv2.cvtColor(output_image_model, cv2.COLOR_BGR2RGB)
    #     yolo_prediction = Image.fromarray(output_image_rgb_model)
    #     predictions.append(yolo_prediction)



    # model yolov8
    yolov8_results = modelv8m(image)
    output_image_model8 = yolov8_results[0].plot()
    output_image_rgb_model8 = cv2.cvtColor(output_image_model8, cv2.COLOR_BGR2RGB)
    yolo8_prediction = Image.fromarray(output_image_rgb_model8)

    return yolo9_prediction, yolo11_prediction, yolo8_prediction
    # return predictions

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image( label="YOLOv11s"),
        gr.Image( label="YOLOv9t"),
        gr.Image( label="YOLOv8m")
    ],
    title="Car make recognition",
    description="Upload a car photo. Car makes that should be detected: Audi, BMW, Chevrolet, Dodge, Ford, Honda, Hyundai, Jeep, Kia, Landrover, Lexus, Mazda, Mercedes, Nissan, Peugeot, Porsche and Toyota"
)
interface.launch()
