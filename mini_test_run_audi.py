import os
import random
import shutil

def create_small_dataset(base_dir, output_dir, target_brand="audi", sample_size=200):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_dir = os.path.join(output_dir, target_brand)
    other_dir = os.path.join(output_dir, "other")

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)

    # Przenieś zdjęcia Audi
    target_path = os.path.join(base_dir, target_brand)
    target_images = []
    for root, _, files in os.walk(target_path):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                target_images.append(os.path.join(root, file_name))

    selected_target = random.sample(target_images, min(len(target_images), sample_size // 2))
    for img_path in selected_target:
        shutil.copy(img_path, target_dir)

    other_brands = [brand for brand in os.listdir(base_dir) if brand != target_brand]
    other_images = []
    for brand in other_brands:
        brand_path = os.path.join(base_dir, brand)
        for root, _, files in os.walk(brand_path):
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    other_images.append(os.path.join(root, file_name))

    selected_other = random.sample(other_images, min(len(other_images), sample_size // 2))
    for img_path in selected_other:
        shutil.copy(img_path, other_dir)

    print(f"Small dataset created in {output_dir}")

# Test
if __name__ == "__main__":
    base_dir = "datasets/cars_make_models"
    output_dir = "datasets/cars_make_models_small"
    create_small_dataset(base_dir, output_dir, target_brand="audi", sample_size=200)
