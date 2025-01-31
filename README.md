# Trained models for car makes recognition
This repository contains a project created for NAI classes. The applicion uses: pretrained on COCO dataset **Yolov5su** model for detecting cars from a photo and trained by us Yolo models (**v5n**, **v8m**, **v9t**, **v11s**) for car make predictions. We used CUDA for a GPU training. Models were trained on 50 epochs on a dataset from kaggle - https://www.kaggle.com/datasets/riotulab/car-make-model-and-generation?select=car-dataset-200, which was divided into smaller and labeled by us. 
# Required libraries to launch the application
Gradio - for interface <br>
```pip install gradio```<br>
Ultralytics - for yolo models <br>
```pip install ultralytics```<br>
# Launching a web application
```python .\NAI_projekt\Nai_cardetection_app\app2.py```
Project summary: <br>
[Sprawozdanie_z_projektu.pdf](https://github.com/user-attachments/files/18588960/Sprawozdanie_z_projektu.pdf)
