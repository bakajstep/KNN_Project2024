import zipfile
from collections import Counter
import os


def count_ner_tags_in_zip(zip_path, tags_of_interest):
    tag_counter = Counter()
    with zipfile.ZipFile(zip_path, 'r') as z:
        for filename in z.namelist():
            with z.open(filename) as file:
                for line in file:
                    line = line.decode('utf-8').strip()
                    if line:
                        parts = line.split()
                        tag = parts[-1]
                        if tag in tags_of_interest:
                            tag_counter[tag] += 1
    return tag_counter


def process_all_zips_in_folder(folder_path):
    tags_of_interest = {
        "O", "B-p", "I-p", "B-i", "I-i", "B-g", "I-g", "B-t", "I-t",
        "B-o", "I-o", "B-m", "I-m", "B-a", "I-a"
    }
    total_counter = Counter()

    # Prochází všechny soubory ve složce
    for file in os.listdir(folder_path):
        if file.endswith('.zip'):
            zip_path = os.path.join(folder_path, file)
            print(f"Processing {zip_path}")
            # Spočítá tagy v každém zip souboru
            zip_counter = count_ner_tags_in_zip(zip_path, tags_of_interest)
            total_counter.update(zip_counter)

    return total_counter


# Cesta ke složce s ZIP soubory
folder_path = '../raw_text/2023-11-23__ner/'
results = process_all_zips_in_folder(folder_path)
print(results)
