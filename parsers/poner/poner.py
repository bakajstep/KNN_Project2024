import os
from datasets import load_dataset
from parsers.util import zip_files, remove_files_by_extension

def num_to_tag(num, tag_names):
    return tag_names[num]

def convert_tags_for_bert(tag):
    mapping = {
        'B-p': 'B-PER', 'I-p': 'I-PER',
        'B-i': 'B-ORG', 'I-i': 'I-ORG',
        'B-g': 'B-LOC', 'I-g': 'I-LOC',
        'B-t': 'B-MISC', 'I-t': 'I-MISC',
        'B-o': 'B-MISC', 'I-o': 'I-MISC',
        'O': 'O'
    }
    return mapping.get(tag, 'O')  # Vrátí 'O' pro nerozpoznané tagy

def dataset_to_conll(dataset, filename, subset, tag_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    with open(full_path, "w", encoding="utf-8") as file:
        for sample in dataset[subset]:
            for token, num_tag in zip(sample["tokens"], sample["ner_tags"]):
                original_tag = num_to_tag(num_tag, tag_names)
                bert_tag = convert_tags_for_bert(original_tag)  # Konverze tagu pro BERT
                file.write(f"{token} {bert_tag}\n")
            file.write("\n")

def prepare_poner(output_dir, dataset_name):
    dataset = load_dataset("romanjanik/PONER")

    tag_names = dataset["train"].features["ner_tags"].feature.names
    dataset_to_conll(dataset, "test.conll", "test", tag_names, output_dir)
    dataset_to_conll(dataset, "train.conll", "train", tag_names, output_dir)
    dataset_to_conll(dataset, "validation.conll", "dev", tag_names, output_dir)

    zip_files(output_dir, os.path.join(output_dir, f"{dataset_name}.zip"), ['.conll'])
    remove_files_by_extension(output_dir, '.conll')