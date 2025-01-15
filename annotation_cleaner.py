import os
# Obsługiwane rozszerzenia plików zdjęć
image_extensions = {".jpg", ".jpeg", ".png"}
base_dir = "cars_make_models_reduced"

for root, dirs, files in os.walk(base_dir):
    # Oddziel pliki na zdjęcia i adnotacje
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    annotations = {os.path.splitext(f)[0] for f in files if f.endswith(".txt")}

    for image in images:
        image_name, _ = os.path.splitext(image)

        # Jeśli brak odpowiadającej adnotacji, usuń zdjęcie
        if image_name not in annotations:
            image_path = os.path.join(root, image)
            print(f"Usuwam zdjęcie bez adnotacji: {image_path}")
            os.remove(image_path)

print("Proces usuwania zakończony.")
