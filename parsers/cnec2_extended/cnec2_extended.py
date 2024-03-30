import os
import zipfile
from io import BytesIO

import requests

from parsers.util import zip_files, remove_files_by_extension


def get_dataset(url_path, output_dir, dataset_name):
    response = requests.get(url_path)
    zip_file = zipfile.ZipFile(BytesIO(response.content))

    root_dir = 'cnec2.0_extended/'
    files_to_extract = ['dev.conll', 'train.conll', 'test.conll']

    # Mapování pouze specifické části anotace
    annotation_mapping = {
        "P": "PER",
        "G": "LOC",
        "I": "ORG"
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in files_to_extract:
        with zip_file.open(root_dir + file_name) as file:  # pylint: disable=R1732
            lines = file.read().decode('utf-8').splitlines()
            processed_lines = []
            line_number = 1
            for line in lines:
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) >= 4:
                        word, lemma, morph, annotation = parts[:4]
                        annotation_parts = annotation.split("-")
                        if len(annotation_parts) == 2:
                            prefix, label = annotation_parts
                            mapped_label = annotation_mapping.get(label, "MISC")
                            mapped_annotation = f"{prefix}-{mapped_label}"
                        else:
                            mapped_annotation = "O" if annotation == "O" else "MISC"
                        processed_lines.append(
                            "\t".join([str(line_number), word, lemma, morph, mapped_annotation]))
                    else:
                        processed_lines.append("\t".join([str(line_number)] + parts))
                else:
                    processed_lines.append("")
                line_number += 1

            # Zápis zpracovaného obsahu do souboru
            output_file_path = os.path.join(output_dir, file_name)
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write("\n".join(processed_lines))

    zip_files(output_dir, os.path.join(output_dir, f"{dataset_name}.zip"), ['.conll'])

    remove_files_by_extension(output_dir, '.conll')


def get_cnec2_extended(url, output_dir, dataset_name):
    get_dataset(url, output_dir, dataset_name)
