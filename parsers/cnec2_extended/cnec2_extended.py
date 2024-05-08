import os
import zipfile
from io import BytesIO, StringIO

import requests
from datasets import Dataset, DatasetDict

from parsers.util import zip_files, remove_files_by_extension

def get_dataset(url_path, output_dir, dataset_name):
    response = requests.get(url_path)
    zip_file = zipfile.ZipFile(BytesIO(response.content))

    root_dir = 'cnec2.0_extended/'
    files_to_extract = ['dev.conll', 'train.conll', 'test.conll']

    annotation_mapping = {
        "P": 1,
        "G": 5,
        "I": 3,
        "T": 7,
        "O": 9,
        "X": 11  # For other unspecified tags
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datasets_dict = {}

    for file_name in files_to_extract:
        with zip_file.open(root_dir + file_name) as file:
            file_content = StringIO(file.read().decode('utf-8'))
            tokens = []
            ner_tags = []
            data = {'tokens': [], 'ner_tags': [], 'sentence_id': []}

            sentence_id = 0  # Initialize sentence counter at zero

            for line in file_content:
                line = line.strip()
                if not line:
                    if tokens:  # Ensure that there are tokens to be added
                        data['tokens'].append(tokens)
                        data['ner_tags'].append(ner_tags)
                        data['sentence_id'].append(sentence_id)
                        tokens = []
                        ner_tags = []
                        sentence_id += 1
                    continue

                parts = line.split('\t')
                if len(parts) >= 4:
                    token = parts[0]
                    ner_tag = parts[3]
                    if '-' in ner_tag:
                        prefix, tag = ner_tag.split('-', 1)
                        ner_tag = annotation_mapping.get(tag, 11) if prefix == "B" else annotation_mapping.get(tag, 11) + 1
                    else:
                        ner_tag = annotation_mapping.get(ner_tag, 11)
                    tokens.append(token)
                    ner_tags.append(ner_tag)

            # Don't forget to add the last sentence
            if tokens:
                data['tokens'].append(tokens)
                data['ner_tags'].append(ner_tags)
                data['sentence_id'].append(sentence_id)

            # Convert to Dataset
            dataset = Dataset.from_dict(data)
            split_name = file_name.replace(".conll", "")
            if split_name == "dev":
                split_name = "validation"  # Rename 'dev' split to 'validation'
            datasets_dict[split_name] = dataset

    # Create DatasetDict and save
    dataset_dict = DatasetDict(datasets_dict)
    dataset_dict.save_to_disk(os.path.join(output_dir, dataset_name))

def get_cnec2_extended(url, output_dir, dataset_name):
    get_dataset(url, output_dir, dataset_name)
