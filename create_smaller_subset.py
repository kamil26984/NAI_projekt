import os
import random
import shutil


def create_reduced_dataset(base_dir, output_dir, sample_size=520):
    """
    Tworzy zmniejszony zestaw danych, losując maksymalnie `sample_size` zdjęć z każdej marki.
    Zdjęcia są kopiowane do folderów z nazwą marki w katalogu `output_dir`.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    brands = [brand for brand in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, brand))]

    for brand in brands:
        brand_path = os.path.join(base_dir, brand)
        brand_output_dir = os.path.join(output_dir, brand)

        os.makedirs(brand_output_dir, exist_ok=True)

        # Zbieranie wszystkich zdjęć danej marki (uwzględniając podfoldery)
        brand_images = []
        for root, _, files in os.walk(brand_path):
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    brand_images.append(os.path.join(root, file_name))

        # Losowy wybór zdjęć
        selected_images = random.sample(brand_images, min(len(brand_images), sample_size))

        # Kopiowanie zdjęć do nowego folderu
        for img_path in selected_images:
            shutil.copy(img_path, brand_output_dir)

    print(f"Reduced dataset created in {output_dir}")


# Test
if __name__ == "__main__":
    base_dir = "datasets/cars_make_models"
    output_dir = "datasets/cars_make_models_reduced"
    create_reduced_dataset(base_dir, output_dir, sample_size=520)
