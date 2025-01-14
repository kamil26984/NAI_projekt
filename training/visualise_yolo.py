import os
import cv2
import random


# Program do wizualizacji wyników detekcji obiektów za pomocą YOLO
# - Wybiera losowo 69 plików z etykietami z katalogu testowego
# - Dla każdego pliku etykiet znajduje odpowiadający mu obraz
# - Rysuje ramki ograniczające na obrazie na podstawie współrzędnych z pliku etykiet
# - Zapisuje obrazy z narysowanymi ramkami do katalogu `visualisation`


NUM_SAMPLES = 69
# Paths
OUTPUT_DIR = os.path.normpath("visualisation")
IMAGES_DIR = os.path.normpath("cars_dataset/test/images")
LABELS_DIR = os.path.normpath("cars_dataset/test/labels")

label_files = os.listdir(LABELS_DIR)
sampled_files = label_files[:NUM_SAMPLES]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to find image path recursively
def find_image_path(image_name, directory):
    for root, _, files in os.walk(directory):
        print(f"Searching in directory: {root}")
        print(f"Files found: {files}")
        if image_name in files:
            return os.path.normpath(os.path.join(root, image_name))
    return None

def draw_bounding_boxes(image_path, label_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # Image dimensions
    print(f"Image size: width={w}, height={h}")

    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # Convert YOLO -> pixels
            x_center *= w
            y_center *= h
            width *= w
            height *= h

            # Debugging
            print(f"YOLO -> x_center={x_center}, y_center={y_center}, width={width}, height={height}")

            # Clamping to image boundaries
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(w - 1, int(x_center + width / 2))
            y2 = min(h - 1, int(y_center + height / 2))

            # Debugging after clamping
            print(f"Pixels -> x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Draw bounding box
            COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # List of colors
            color = COLORS[int(class_id) % len(COLORS)]  # Select color based on class_id
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Process selected files
for label_file in sampled_files:
    image_name = label_file.replace('.txt', '.jpg')
    image_path = find_image_path(image_name, IMAGES_DIR)
    label_path = os.path.normpath(os.path.join(LABELS_DIR, label_file))

    # Debugging paths
    print(f"Processing label file: {label_file}")
    print(f"Expected image name: {image_name}")
    print(f"Image path: {image_path}")
    print(f"Label path: {label_path}")

    if image_path:  # If image found
        result_image = draw_bounding_boxes(image_path, label_path)
        output_path = os.path.normpath(os.path.join(OUTPUT_DIR, os.path.basename(image_name)))
        cv2.imwrite(output_path, result_image)
        print(f"Saved image: {output_path}")
    else:
        print(f"Image not found: {image_name}")

print("Visualization completed. Check images in folder:", OUTPUT_DIR)