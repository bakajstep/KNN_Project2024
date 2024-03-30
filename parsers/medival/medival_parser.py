import io
import os
import zipfile

import requests

from parsers.util import zip_files, remove_files_by_extension


def prepare_medival(url, output_dir='.', dataset_name="conllu_dataset.zip"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))  # pylint: disable=R1732
    z.extractall(path=output_dir)

    # Získání seznamu všech rozbalených souborů
    unzipped_files = os.listdir(output_dir)

    # Filtrace souborů, které končí na .sentence.txt a jejich zpracování
    for file in unzipped_files:
        if file.endswith('.sentences.txt'):
            base_name = file[:-len('.sentences.txt')]
            text_file_path = os.path.join(output_dir, file)
            annotations_file_path = os.path.join(output_dir, f"{base_name}.ner_tags.txt")

            # Zkontrolujte, zda existuje odpovídající soubor s anotacemi
            if os.path.exists(annotations_file_path):
                # Vytvoření jména výstupního souboru s příponou .conllu
                conllu_filename = os.path.join(output_dir, f"{base_name}.conll")

                # Zpracujte soubory
                process_files(text_file_path, annotations_file_path, conllu_filename)
            else:
                print(f"Nenalezen odpovídající soubor s anotacemi pro {file}")

    zip_files(output_dir, os.path.join(output_dir, f"{dataset_name}.zip"), ['.conll'])

    remove_files_by_extension(output_dir, '.txt')
    remove_files_by_extension(output_dir, '.conll')
    remove_files_by_extension(output_dir, '.docx')


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
