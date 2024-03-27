import fnmatch
import os
import tempfile
import zipfile

import argparse
from pagexml.parser import parse_pagexml_file
from pagexml.helper.pagexml_helper import make_text_region_text
from transformers import pipeline


# https://github.com/roman-janik/diploma_thesis_program/blob/a23bfaa34d32f92cd17dc8b087ad97e9f5f0f3e6/train_ner_model.py#L28
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model directory path.')
    # parser.add_argument('--results_csv', required=True, help='Results CSV file.')
    args = parser.parse_args()
    return args

def extract_sentences_from_pagexml(xml_file):
    doc = parse_pagexml_file(xml_file)

    lines = doc.get_lines()
    regions = make_text_region_text(lines)

    return regions[0]


def find_zip_files(directory):
    zip_files = []
    for root, dirs, files in os.walk(directory):
        for file in fnmatch.filter(files, '*.zip'):
            zip_files.append(os.path.join(root, file))
    return zip_files


def process_zip_files(file_name):
    sentences = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            files = [f for f in os.listdir(temp_dir) if f.endswith('.xml')]
            for file in files:
                file_path = os.path.join(temp_dir, file)
                sentences.append(extract_sentences_from_pagexml(file_path))

    return sentences


def prediction_to_conll(out):
    conll_output = []
    current_sentence = []

    for item in out:
        word = item["word"]
        entity_tag = 'O' if item['entity_group'] == 'LABEL_0' else item['entity_group'].replace("LABEL_", "B-")

        # TODO change to format we use in our solution
        current_sentence.append(f"{word}\t_\t_\t{entity_tag}")

        if word.endswith('.'):
            conll_output.append('\n'.join(current_sentence))
            conll_output.append("")
            current_sentence = []

    if current_sentence:
        conll_output.append('\n'.join(current_sentence))

    return conll_output


def main():
    args = parse_arguments()

    model_checkpoint = args.model

    # path to directory with pagexml zip files
    xml_files_directory = "./pageXml"
    zip_files = find_zip_files(xml_files_directory)
    for file_name in zip_files:
        sentences = process_zip_files(file_name)
        # path to model current model
        # model_checkpoint = "./models/2024-03-25-14-07-baseline_linear_lr_5e5_5_epochs-h/model"
        token_classifier = pipeline(
            "token-classification", model=model_checkpoint, aggregation_strategy="simple"
        )
        predictions = token_classifier(sentences[0])
        conll = prediction_to_conll(predictions)
        output_file_path = file_name[:-4].replace('/', '_').replace(".", "") + ".conll"
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for c in conll:
                f.write(c)


if __name__ == "__main__":
    main()
