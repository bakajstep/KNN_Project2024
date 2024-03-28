from datasets import load_dataset
import os


def num_to_tag(num, tag_names):
    return tag_names[num]


def dataset_to_conll(dataset, filename, subset, tag_names):
    output_dir = "ourdatasets/wikiann"
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, filename)

    with open(full_path, "w", encoding="utf-8") as file:
        for sample in dataset[subset]:
            for token, num_tag in zip(sample["tokens"], sample["ner_tags"]):
                tag = num_to_tag(num_tag, tag_names)
                file.write(f"{token} {tag}\n")
            file.write("\n")


def prepare_wikiann():
    dataset = load_dataset("wikiann", "cs")

    tag_names = dataset["train"].features["ner_tags"].feature.names
    dataset_to_conll(dataset, "test.conll", "test", tag_names)
    dataset_to_conll(dataset, "train.conll", "train", tag_names)
    dataset_to_conll(dataset, "validation.conll", "validation", tag_names)
