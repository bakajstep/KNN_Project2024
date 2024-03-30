import argparse
import fnmatch
import os
import tempfile
import zipfile

from pagexml.helper.pagexml_helper import make_text_region_text
from pagexml.parser import parse_pagexml_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch

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
    previous_tag = None

    for item in out:
        word = item["word"]
        entity_group = item['entity_group']

        if entity_group == 'LABEL_0':
            entity_tag = 'O'
            previous_tag = None
        else:
            entity_type = entity_group.replace("LABEL_", "")
            if previous_tag == entity_type:
                entity_tag = f"I-{entity_type}"
            else:
                entity_tag = f"B-{entity_type}"
            previous_tag = entity_type

        current_sentence.append(f"{word}\t_\t_\t{entity_tag}")

        if word.endswith('.'):
            conll_output.append('\n'.join(current_sentence))
            conll_output.append("")
            current_sentence = []
            previous_tag = None

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
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        text = sentences[0]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            predictions = model(**inputs)
        conll = prediction_to_conll(predictions)
        output_file_path = file_name[:-4].replace('/', '_').replace(".", "") + ".conll"

        print("Output file:", output_file_path)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for c in conll:
                f.write(c)


if __name__ == "__main__":
    main()
