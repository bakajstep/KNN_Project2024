import os

from datasets import load_dataset

from parsers.util import zip_files, remove_files_by_extension


def num_to_tag(num, tag_names):
    return tag_names[num]


def dataset_to_conll(dataset, filename, subset, tag_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, filename)

    with open(full_path, "w", encoding="utf-8") as file:
        row_number = 0
        for sample in dataset[subset]:
            for token, num_tag in zip(sample["tokens"], sample["ner_tags"]):
                tag = num_to_tag(num_tag, tag_names)
                file.write(f"{row_number}\t{token}\t{tag}\n")
                row_number += 1
            file.write("\n")


def prepare_wikiann(output_dir, dataset_name):
    dataset = load_dataset("wikiann", 'cs')

    tag_names = dataset["train"].features["ner_tags"].feature.names
    dataset_to_conll(dataset, "test.conll", "test", tag_names, output_dir)
    dataset_to_conll(dataset, "train.conll", "train", tag_names, output_dir)
    dataset_to_conll(dataset, "validation.conll", "validation", tag_names, output_dir)

    zip_files(output_dir, os.path.join(output_dir, f"{dataset_name}.zip"), ['.conll'])

    remove_files_by_extension(output_dir, '.conll')
