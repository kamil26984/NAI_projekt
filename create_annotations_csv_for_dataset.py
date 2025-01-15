import os
import csv


# program generuje plik csv z anotacjami do zestawu danych cars_make_models
# program wybiera markę z nazwy folderu, w którym znajduje się zdjęcie

#zmienić na to jak kto ma ułożone datasety u siebie, wskazuje na główny folder setu
DATASETS_DIR = r"C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\cars_make_models_reduced"



def generate_annotations(base_dir, output_file):
    """
    Generuje plik CSV z anotacjami do zestawu danych gdzie marka jest wyciągana z nazwy folderu.
    """
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


def generate_annotations_for_reduced_dataset(base_dir, output_csv):
    with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(output_csv).st_size == 0:
            writer.writerow(["image_path", "brand"])  # Nagłówek

        images_processed = 0  # Licznik przetworzonych obrazów

        for root, _, files in os.walk(base_dir):
            print(f"Checking directory: {root}")  # Debugowanie katalogów
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Pełna ścieżka do pliku
                    image_path = os.path.join(root, file_name).replace("\\", "/")
                    print(f"Found file: {image_path}")  # Debugowanie znalezionych plików

                    # Wyciąganie marki z nazwy pliku
                    brand = file_name.split('-')[0].lower()
                    print(f"Detected brand: {brand}")  # Debugowanie marki

                    # Zapis do pliku CSV
                    writer.writerow([image_path, brand])
                    images_processed += 1

        print(f"Processed {images_processed} images.")
        print(f"Annotations saved in {output_csv}")


if __name__ == "__main__":
    test_cars = ["audi", "bmw", "mercedes", "toyota"]
    all_cars = []
    for name in os.listdir(DATASETS_DIR):
        if os.path.isdir(os.path.join(DATASETS_DIR, name)):
            all_cars.append(name)

    print(all_cars)

    output_file = DATASETS_DIR + "cars_make_models_reduced/annotations_test.csv"
    output_csv = "cars_make_models_reduced/annotations.csv"



    for car in all_cars:
        base_dir = DATASETS_DIR+"/"+car
        #generate_annotations(base_dir, output_file)
        generate_annotations_for_reduced_dataset(base_dir, output_csv)
