import os


def simplify_labels(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Zastąp wszystkie class_id przez 0
            new_lines = []
            for line in lines:
                parts = line.split()
                parts[0] = "0"  # Zmieniamy class_id na 0
                new_lines.append(" ".join(parts))

            # Zapisz zmienione etykiety do tego samego pliku
            with open(file_path, "w") as f:
                f.writelines("\n".join(new_lines) + "\n")


# Ścieżki do folderów
train_dir = "yolo_format/train"
val_dir = "yolo_format/val"

# Uprość etykiety w obu folderach
simplify_labels(train_dir)
simplify_labels(val_dir)

print("Wszystkie klasy zostały zamienione na jedną kategorię: 'vehicle'.")
