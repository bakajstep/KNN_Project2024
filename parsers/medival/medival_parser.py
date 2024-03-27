import glob
import zipfile
import io
import os
import requests


def zip_files(files, zip_name, path):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))

    for f in glob.glob("{}dataset_*".format(path)):
        os.remove(f)


def unzip_and_process(url, extract_to='.', name_dataset="conllu_dataset.zip"):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=extract_to)

    conllu_files = []  # Seznam souborů pro zazipování

    # Získání seznamu všech rozbalených souborů
    unzipped_files = os.listdir(extract_to)

    # Filtrace souborů, které končí na .sentence.txt a jejich zpracování
    for file in unzipped_files:
        if file.endswith('.sentences.txt'):
            base_name = file[:-len('.sentences.txt')]
            text_file_path = os.path.join(extract_to, file)
            annotations_file_path = os.path.join(extract_to, f"{base_name}.ner_tags.txt")

            # Zkontrolujte, zda existuje odpovídající soubor s anotacemi
            if os.path.exists(annotations_file_path):
                # Vytvoření jména výstupního souboru s příponou .conllu
                conllu_filename = os.path.join(extract_to, f"{base_name}.conllu")

                # Zpracujte soubory
                process_files(text_file_path, annotations_file_path, conllu_filename)

                # Přidání do seznamu pro zazipování
                conllu_files.append(conllu_filename)
            else:
                print(f"Nenalezen odpovídající soubor s anotacemi pro {file}")

    # Zazipování a odstranění souborů
    if conllu_files:
        zip_name = os.path.join(extract_to, name_dataset)
        zip_files(conllu_files, zip_name, extract_to)


def process_files(text_file_path, annotations_file_path, output_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as text_file, \
            open(annotations_file_path, 'r', encoding='utf-8') as annotations_file, \
            open(output_file_path, 'w', encoding='utf-8') as output_file:

        for line_text, line_annotations in zip(text_file, annotations_file):
            tokens = line_text.strip().split()
            annotations = line_annotations.strip().split()

            for token, annotation in zip(tokens, annotations):
                output_file.write(f"{token}\t_\t_\t_\t_\t_\t_\t_\t{annotation}\n")

            output_file.write("\n")  # Konec věty označený novým řádkem

# Ukázka použití
# url = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5024/named-entity-recognition-annotations-small.zip?sequence=2&isAllowed=y'
# extract_to = './zip/'
# unzip_and_process(url, extract_to)
