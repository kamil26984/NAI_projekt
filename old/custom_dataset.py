import os
from torch.utils.data import Dataset
from PIL import Image

# Ten program definiuje niestandardowy zbiór danych dla PyTorch, który ładuje obrazy i odpowiadające im etykiety z podanych katalogów.
# Wynikiem działania programu jest obiekt `CustomDataset`, który można użyć do trenowania modeli w PyTorch.

#szczerze mówiąc nie pamiętam ską to jest i po co ale boje się usuwać bo nigdy nie wiadomo w programowaniu

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.transform = transform

        # Pobierz listę obrazów
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.image_files.sort()  # Dla spójności
        self.label_files = [os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt") for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        label_path = self.label_files[idx]
        with open(label_path, "r") as f:
            labels = [line.strip().split() for line in f.readlines()]
            labels = [list(map(float, label)) for label in labels]

        if self.transform:
            image = self.transform(image)

        return image, labels
