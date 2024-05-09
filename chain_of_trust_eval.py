import argparse
import time
import logging
from datetime import datetime

import datasets
import evaluate
from transformers import pipeline
import numpy as np
from yaml import safe_load

from inference_new_model import process_text


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    # parser.add_argument('--results_csv', required=True, help='Results CSV file.')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as file:
        return safe_load(file)


def log_msg(msg: str):
    print(msg)
    logging.info(msg)


def create_pipeline(model_path):
    return pipeline("ner", model=model_path, aggregation_strategy="simple")


def prepare_datasets(config: dict):
    raw_datasets = {key: datasets.load_from_disk(value["path"]) for (key, value) in config["datasets"].items()}
    label_names = ['O', 'B-p', 'I-p', 'B-i', 'I-i', 'B-g', 'I-g', 'B-t', 'I-t', 'B-o', 'I-o']

    if "test" not in config["datasets"]:
        config["datasets"]["test"] = {
            "name": "Combined Test Dataset",
            "desc": "A combination of various splits from multiple datasets.",
            "path": "path/to/combined_dataset"  # Tento údaj je ilustrativní
        }

    concat_datasets = datasets.DatasetDict({
        "test": datasets.concatenate_datasets(
            [dataset[split] for dataset in raw_datasets.values() for split in dataset if
             split in ['train', 'test', 'validation']]
        )
    })

    return label_names, concat_datasets


def evaluate_predictions(predictions, references, metric_evaluator):
    # Vyhodnocení predikcí
    return metric_evaluator.compute(predictions=predictions, references=references)


def extract_tags_from_prediction(prediction):
    # Předpokládáme, že 'prediction' je seznam stringů, kde každý string obsahuje jednu větu ve formátu "slovo tag"
    # Rozdělíme každou větu na slova, a pak extrahujeme pouze tagy
    lines = prediction.split('\n')
    tags = [line.split()[1] if len(line.split()) > 1 else 'O' for line in lines if line.strip()]
    return tags


def main():
    start_time = time.monotonic()
    args = parse_arguments()
    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)
    model1 = create_pipeline(config['models']['model1']['path'])
    model2 = create_pipeline(config['models']['model2']['path'])
    model3 = create_pipeline(config['models']['model3']['path'])

    label_names, test_dataset = prepare_datasets(config)

    metric_evaluator = evaluate.load("seqeval")
    all_results = []

    test_dataset = test_dataset['test']

    for example in test_dataset:
        text = " ".join(example['tokens'])
        prediction_output = process_text([text], model1, model2, model3)
        predictions = extract_tags_from_prediction(prediction_output[0])

        # Získání anotací pro porovnání
        references = [label_names[tag_idx] for tag_idx in example['ner_tags']]

        # Evaluační metrika
        results = evaluate_predictions([predictions], [references], metric_evaluator)
        all_results.append(results)

    # Průměrování výsledků
    avg_results = np.mean(all_results)
    print(
        f"Average F1 Score: {avg_results['f1']}, Average Precision: {avg_results['precision']}, Average Recall: {avg_results['recall']}")

    end_time = time.monotonic()
    log_msg("Elapsed script time: {}\n".format(datetime.timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()
