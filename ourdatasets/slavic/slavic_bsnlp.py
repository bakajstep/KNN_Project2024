import os
import zipfile
from io import BytesIO

import requests


def load_annotations(zip_file, path):
    annotations = {}
    with zip_file.open(path) as f:
        for line in f:
            line_decoded = line.decode('utf-8').strip()
            parts = line_decoded.split('\t')
            if len(parts) >= 4:
                annotations[parts[0]] = parts[2]
    return annotations


def load_raw_text(zip_file, path):
    with zip_file.open(path) as f:
        return f.read().decode('utf-8')


def convert_to_conll(annotations, raw_text):
    conll_lines = []
    sentences = raw_text.split('. ')
    for sentence in sentences:
        tokens = sentence.split()
        for token in tokens:
            if token.endswith('.'):
                token = token[:-1]
            tag = annotations.get(token, 'O')
            conll_lines.append(f"{token}\t{tag}")
        conll_lines.append("")
    return "\n".join(conll_lines)


def prepare_slavic(zip_url, output_file_name, language_code):
    response = requests.get(zip_url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))

    annotated_dir_prefix = f"training_pl_cs_ru_bg_rc1/annotated/{language_code}/"
    raw_dir_prefix = f"training_pl_cs_ru_bg_rc1/raw/{language_code}/"
    output_dir = f"ourdatasets/slavic/{language_code}/"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, output_file_name), 'w', encoding='utf-8') as output_file:
        for zip_info in zip_file.infolist():
            if zip_info.filename.startswith(annotated_dir_prefix) and zip_info.filename.endswith('.out'):
                raw_filename = zip_info.filename.replace(annotated_dir_prefix, raw_dir_prefix).replace('.out', '.txt')

                annotations = load_annotations(zip_file, zip_info.filename)
                raw_text = load_raw_text(zip_file, raw_filename)
                conll_content = convert_to_conll(annotations, raw_text)

                output_file.write(conll_content + "\n")
