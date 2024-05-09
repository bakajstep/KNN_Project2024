import os
from datasets import load_dataset

def num_to_tag(num, tag_names):
    return tag_names[num]

def dataset_to_conll(dataset, filename, subset, tag_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename + ".csv")

    with open(full_path, "w", encoding="utf-8") as file:
        file.write('"ID","Word","Lemma","Morphology","Annotation"\n')  # CSV header
        token_counter = 1
        for sample in dataset[subset]:
            for token, num_tag in zip(sample["tokens"], sample["ner_tags"]):
                tag = num_to_tag(num_tag, tag_names)
                file.write(f'"{token_counter}","{token}","{token}","_","{tag}"\n')  # Simplified CSV output
                token_counter += 1
            file.write("\n")

def prepare_poner(output_dir, dataset_name):
    dataset = load_dataset("romanjanik/PONER")

    tag_names = dataset["train"].features["ner_tags"].feature.names
    dataset_to_conll(dataset, "test", "test", tag_names, output_dir)
    dataset_to_conll(dataset, "train", "train", tag_names, output_dir)
    dataset_to_conll(dataset, "validation", "dev", tag_names, output_dir)

    from parsers.util import zip_files, remove_files_by_extension
    zip_files(output_dir, os.path.join(output_dir, f"{dataset_name}.zip"), ['.csv'])
    remove_files_by_extension(output_dir, '.csv')

# Použití příkladu
output_directory = "output_directory"
dataset_name = "poner_dataset"
prepare_poner(output_directory, dataset_name)
