import os
from pathlib import Path


#Program do zmiany różnych klas pojazdów na jedną wspólną, był do innego datasetu i innego pomysłu ale może jakaś część kodu jeszcze się nada

def remove_empty_lines(dataset_dir):
    """
    Usuwa puste linie z plików .txt w podanym katalogu.
    """
    annotation_files = list(Path(dataset_dir).glob("*.txt"))
    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as file:
            lines = file.readlines()

        # Filtrujemy tylko niepuste linie
        non_empty_lines = [line for line in lines if line.strip()]

        # Zapisujemy plik na nowo
        with open(annotation_file, 'w') as file:
            file.writelines(non_empty_lines)

        print(f"Wyczyszczono plik: {annotation_file.name}")

# Użycie
dataset_dir = "datasets/vriv"  # Ścieżka do katalogu z plikami .txt
remove_empty_lines(dataset_dir)


#dataset_dir = "datasets/vriv"  # Podaj ścieżkę katalogu

files = list(Path(dataset_dir).glob("*"))
print(f"Znaleziono {len(files)} plików w katalogu {dataset_dir}:")
for f in files:
    print(f.name)
    print(f"Plik: {f.name}, rozszerzenie: {f.suffix}")


def simplify_labels(dataset_dir, target_class=0):
    """
    Zmienia wszystkie klasy w plikach .txt na jedną zbiorczą klasę (domyślnie 0),
    zachowując pierwszą linię (nagłówki), a pozostałe przetwarzając.
    """
    annotation_files = list(Path(dataset_dir).glob("*.txt"))

    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as file:
            lines = file.readlines()

        # Zachowujemy pierwszą linię
        if len(lines) > 0:
            first_line = lines[0]
            data_lines = lines[1:]
        else:
            first_line = ""
            data_lines = []

        simplified_lines = [first_line]

        for idx, line in enumerate(data_lines, start=2):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"[DEBUG] Pomijam niepoprawną linię w pliku {annotation_file.name} (linia {idx}): {line.strip()}")
                continue

            parts[0] = str(target_class)
            simplified_lines.append(" ".join(parts) + "\n")

        if simplified_lines:
            with open(annotation_file, 'w') as file:
                file.writelines(simplified_lines)
            print(f"Zmodyfikowano plik: {annotation_file.name}")
        else:
            print(f"Plik {annotation_file.name} jest pusty lub zawiera tylko niepoprawne linie.")

    print(f"Przetworzono {len(annotation_files)} plików .txt w katalogu {dataset_dir}")


dataset_dir = "datasets/vriv"
simplify_labels(dataset_dir)
