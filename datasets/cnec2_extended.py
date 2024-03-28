import zipfile
from io import BytesIO

from conllu import parse

import requests


def get_dataset(url_path):
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

    file_contents = {}

    for file_name in files_to_extract:
        with zip_file.open(root_dir + file_name) as file:
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
                            mapped_label = annotation_mapping.get(label, label)
                            mapped_annotation = f"{prefix}-{mapped_label}"
                        else:
                            mapped_annotation = annotation
                        processed_lines.append("\t".join([str(line_number), word, lemma, morph, mapped_annotation]))
                    else:
                        processed_lines.append("\t".join([str(line_number)] + parts))
                else:
                    processed_lines.append("")
                line_number += 1
            file_contents[file_name] = "\n".join(processed_lines)

    return file_contents


def get_cnec2_extended(url):
    return get_dataset(url)
