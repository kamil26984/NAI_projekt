# Trained models for car makes recognition
This repository contains a project created for NAI classes. The applicion uses: pretrained **Yolov5su** model for detecting cars from a photo and trained by us Yolo models (**v5n**, **v8m**, **v9t**, **v11s**) for car make predictions. We used CUDA for a GPU training. Models were trained on 50 epochs.
# Required libraries to launch the application
Gradio - for interface <br>
```pip install gradio```<br>
Ultralytics - for yolo models <br>
```pip install ultralytics```<br>
Pillow – images proccessing <br>
```pip install pillow``` <br>
OpenCV – images proccessing <br>
```pip install opencv-python``` <br>
NumPy – arrays operations <br>
```pip install numpy```
