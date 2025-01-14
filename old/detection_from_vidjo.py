import cv2
import os
from ultralytics import YOLO

#program demonstrujący, ze mój wytrenowany model yolo radzi sobie z wykrywaniem samochodów na filmach
#program zapisuje wykryte samochody do folderu output/cars
#program działa na filmach z folderu datasets/vriv_divided/minitest
#już się pewnie nie przyda ale dalej jestem dumny z mojego yolo

if not os.path.exists("output/cars"):
    os.makedirs("output/cars")

def detect_and_save_cars(video_path, model, output_dir, conf_threshold=0.5):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Couldn't open video {video_path}")
        return

    frame_count = 0
    car_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame)

        for r in results:
            for det in r.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                conf = det.conf[0]
                cls = int(det.cls[0])

                print(f"Detection: cls={cls}, conf={conf}")

                if conf > conf_threshold and cls == 1 or cls == 2:
                    cropped_car = frame[y1:y2, x1:x2]
                    car_filename = f"{output_dir}/car_{frame_count}_{car_count}.jpg"
                    cv2.imwrite(car_filename, cropped_car)
                    car_count += 1

    cap.release()
    print(f"Saved {car_count} cars from {frame_count} frames")

if __name__ == "__main__":
    model = YOLO("yolo11s.pt")
    video_dir = "datasets/vriv_divided/minitest"
    output_dir = "output/cars"

    # Iteruj tylko przez pliki wideo
    for video_file in os.listdir(video_dir):
        if video_file.lower().endswith(('.mp4', '.mov', '.avi')):
            video_path = os.path.join(video_dir, video_file)
            print(f"Processing video: {video_file}")
            detect_and_save_cars(video_path, model, output_dir)
