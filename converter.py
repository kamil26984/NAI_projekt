import os
import scipy.io
import cv2

# Ścieżki do datasetu
mat_file_path = "stanford_cars/devkit/cars_train_annos.mat"
images_dir = "stanford_cars/cars_train"
output_dir = "yolo_format"

# Wczytanie pliku .mat
data = scipy.io.loadmat(mat_file_path)
annotations = data['annotations'][0]  # Pobieramy wszystkie adnotacje

# Tworzymy folder wyjściowy
os.makedirs(output_dir, exist_ok=True)

# Funkcja skalowania bounding boxów
def scale_bbox(bbox, orig_width, orig_height, new_width, new_height):
    x_scale = new_width / orig_width
    y_scale = new_height / orig_height
    x1, y1, x2, y2 = bbox
    return [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale]

# Przetwarzanie adnotacji
for i, anno in enumerate(annotations):
    try:
        # Odczytanie danych
        filename = anno['fname'][0]  # Nazwa pliku obrazu
        bbox_x1 = float(anno['bbox_x1'][0][0])
        bbox_y1 = float(anno['bbox_y1'][0][0])
        bbox_x2 = float(anno['bbox_x2'][0][0])
        bbox_y2 = float(anno['bbox_y2'][0][0])
        class_id = int(anno['class'][0][0]) - 1  # Klasy w YOLO zaczynają się od 0

        # Wczytanie obrazu
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue

        orig_height, orig_width = img.shape[:2]

        # Konwersja obrazu do wymiarów 640x640
        new_width, new_height = 640, 640
        resized_img = cv2.resize(img, (new_width, new_height))
        cv2.imwrite(os.path.join(output_dir, filename), resized_img)

        # Skalowanie bounding boxów
        scaled_bbox = scale_bbox([bbox_x1, bbox_y1, bbox_x2, bbox_y2], orig_width, orig_height, new_width, new_height)

        # Konwersja do formatu YOLO: x_center, y_center, width, height
        x_center = (scaled_bbox[0] + scaled_bbox[2]) / 2 / new_width
        y_center = (scaled_bbox[1] + scaled_bbox[3]) / 2 / new_height
        bbox_width = (scaled_bbox[2] - scaled_bbox[0]) / new_width
        bbox_height = (scaled_bbox[3] - scaled_bbox[1]) / new_height

        # Zapis pliku .txt w formacie YOLO
        yolo_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(output_dir, yolo_filename), "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    except Exception as e:
        print(f"Error processing {i}: {e}")

print("Processing complete.")
