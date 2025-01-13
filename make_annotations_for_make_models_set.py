import os
import csv


#zmienić na to jak kto ma ułożone datasety u siebie, wskazuje na główny folder setu
DATASETS_DIR = "datasets/cars_make_models"

def generate_annotations(base_dir, output_file):
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.stat(output_file).st_size == 0:
            writer.writerow(["image_path", "brand"])

        for root, dirs, files in os.walk(base_dir):
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file_name).replace("\\", "/")
                    last_folder = os.path.basename(os.path.dirname(image_path))
                    brand = last_folder.split('-')[0]
                    writer.writerow([image_path, brand])


    print(f"Annotations saved to {output_file}")

if __name__ == "__main__":
    test_cars = ["audi", "bmw", "mercedes", "toyota"]
    all_cars = []
    for name in os.listdir(DATASETS_DIR):
        if os.path.isdir(os.path.join(DATASETS_DIR, name)):
            all_cars.append(name)

    output_file = DATASETS_DIR + "/annotations_test.csv"

    for car in all_cars:
        base_dir = f"DATASETS_DIR/{car}"
        generate_annotations(base_dir, output_file)
