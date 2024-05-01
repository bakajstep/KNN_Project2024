import os
import zipfile
from io import BytesIO

import requests

from parsers.util import zip_files, remove_files_by_extension


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
    line_number = 0
    for sentence in sentences:
        tokens = sentence.split()
        prev_tag = 'O'
        for token in tokens:
            if token.endswith('.'):
                token = token[:-1]
            tag = annotations.get(token, 'O')
            bio_tag = 'B-' + tag if tag != 'O' and (
                    prev_tag == 'O' or prev_tag != tag) else 'I-' + tag if tag != 'O' else 'O'
            conll_lines.append(f"{line_number}\t{token}\t_\t_\t_\t_\t_\t_\t_\t{bio_tag}")
            prev_tag = tag
            line_number += 1
        conll_lines.append("")
    return "\n".join(conll_lines)


def prepare_datasets(zip_url, output_file_name, annotated_dirs, raw_dirs, output_dir):
    response = requests.get(zip_url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))  # pylint: disable=R1732

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, output_file_name), 'w', encoding='utf-8') as output_file:
        for annotated_dir, raw_dir in zip(annotated_dirs, raw_dirs):
            for zip_info in zip_file.infolist():
                if zip_info.filename.startswith(annotated_dir) and zip_info.filename.endswith(
                        '.out'):
                    raw_filename = zip_info.filename.replace(annotated_dir, raw_dir).replace('.out',
                                                                                             '.txt')

                    annotations = load_annotations(zip_file, zip_info.filename)
                    raw_text = load_raw_text(zip_file, raw_filename)
                    conll_content = convert_to_conll(annotations, raw_text)

                    output_file.write(conll_content + "\n")


def prepare_slavic(zip_url_training, zip_url_validation, output_dir, dataset_name):
    prepare_datasets(
        zip_url_training,
        "train.conll",
        ["bsnlp2021_train_r1/annotated/asia_bibi/cs/",
         "bsnlp2021_train_r1/annotated/brexit/cs/",
         "bsnlp2021_train_r1/annotated/nord_stream/cs/",
         "bsnlp2021_train_r1/annotated/ryanair/cs/",
         "bsnlp2021_train_r1/annotated/other/cs/"],
        ["bsnlp2021_train_r1/raw/asia_bibi/cs/",
         "bsnlp2021_train_r1/raw/brexit/cs/",
         "bsnlp2021_train_r1/raw/nord_stream/cs/",
         "bsnlp2021_train_r1/raw/ryanair/cs/",
         "bsnlp2021_train_r1/raw/other/cs/"],
        output_dir
    )
    prepare_datasets(
        zip_url_validation,
        "test.conll",
        ["annotated_corrected/covid-19/cs/", "annotated_corrected/us_election_2020/cs/"],
        ["raw/covid-19/cs/", "raw/us_election_2020/cs/"],
        output_dir
    )
    zip_files(output_dir, os.path.join(output_dir, f"{dataset_name}.zip"), ['.conll'])

    remove_files_by_extension(output_dir, '.txt')
    remove_files_by_extension(output_dir, '.conll')
