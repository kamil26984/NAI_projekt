import os

# Ścieżki do folderów z plikami etykiet (train, val, test)
folders = [
    r'C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\divided_set\train',
    r'C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\divided_set\val',
    r'C:\Users\Admin\Desktop\NAI PROEJKT\NAI_projekt\divided_set\test'
]

# Iteracja po wszystkich folderach
for folder in folders:
    # Iteracja po wszystkich plikach w folderze
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder, filename)

            # Otwórz plik etykiety
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Popraw format (usuń szóstą kolumnę)
            new_lines = []
            for line in lines:
                parts = line.split()
                # Upewnij się, że linia zawiera dokładnie 6 kolumn
                if len(parts) == 6:
                    parts = parts[:-1]  # Usuwamy ostatnią (szóstą) kolumnę
                new_lines.append(" ".join(parts) + '\n')

            # Zapisz poprawiony plik etykiety
            with open(file_path, 'w') as file:
                file.writelines(new_lines)

            print(f"Poprawiono etykiety w pliku: {filename} w folderze {folder}")
