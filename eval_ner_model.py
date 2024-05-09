# Author: Roman Janík
# Script for training baseline model from RobeCzech on CNEC 2.0 CoNNL and CHNEC 1.0 datasets.
#
# Partially taken over from Hugging Face Course Chapter 7 Token classification:
# https://huggingface.co/course/en/chapter7/2?fw=pt
#

import argparse
import csv
import datetime
import logging
import os
import time

import datasets
import transformers
import torch
import evaluate
import numpy as np
import pandas as pd

from accelerate import Accelerator
from tqdm.auto import tqdm
from yaml import safe_load
#from torch.utils.tensorboard import SummaryWriter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    #parser.add_argument('--results_csv', required=True, help='Results CSV file.')
    args = parser.parse_args()
    return args


def log_msg(msg: str):
    print(msg)
    logging.info(msg)


def log_summary(exp_name: str, config: dict):
    log_msg("{:<24}{}\n{:<24}{}".format(
        "Name:", exp_name.removeprefix("exp_configs_ner/").removesuffix(".yaml"), "Description:", config["desc"]))
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n".format(
        "Start time:", ct, "Model:", config["model"]["name"],
        "Datasets:", [dts["name"] for dts in config["datasets"].values()]))

    cf_t = config["training"]
    log_msg("Parameters:\n{:<24}{}\n{:<24}{}\n{:<24}{}".format(
        "Num train epochs:", cf_t["num_train_epochs"], "Batch size:", cf_t["batch_size"],
        "Val batch size:", cf_t["batch_size"]))
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n{:<24}{}".format(
        "Learning rate:", cf_t["optimizer"]["learning_rate"], "Weight decay:", cf_t["optimizer"]["weight_decay"],
        "Lr scheduler:",
        cf_t["lr_scheduler"]["name"], "Warmup ratio:", cf_t["lr_scheduler"]["num_warmup_steps"]))


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

# TODO upravit aby pracovalo pouze s test datasetem
def prepare_datasets(config: dict):
    raw_datasets = {key: datasets.load_from_disk(value["path"]) for (key, value) in config["datasets"].items()}
    label_names = ['O', 'B-p', 'I-p', 'B-i', 'I-i', 'B-g', 'I-g', 'B-t', 'I-t', 'B-o', 'I-o']

    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"]["path"], add_prefix_space=True)

    concat_datasets = datasets.DatasetDict({
        "test": datasets.concatenate_datasets(
            [dataset[split] for dataset in raw_datasets.values() for split in dataset if
             split in ['train', 'test', 'validation']]
        )
    })

    return tokenizer, label_names, concat_datasets


# noinspection PyArgumentList
def main():
    start_time = time.monotonic()
    output_dir = "../results"
    model_dir = "../results/model"
    log_dir = "../results/logs"
    args = parse_arguments()

    datasets_dir = "../results/datasets"

    # Load config file
    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)

    # Start logging, print experiment configuration
    logging.basicConfig(filename=os.path.join(output_dir, "experiment_results.txt"), level=logging.INFO,
                        encoding='utf-8', format='%(message)s')
    #log_msg("Experiment summary:\n")
    #log_summary(args.config, config)
    #log_msg("-" * 80 + "\n")

    # Init tensorboard writer
    #writer = SummaryWriter(log_dir)

    tokenizer, label_names, test_datasets = prepare_datasets(config)
    data_collator = transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    model = transformers.AutoModelForTokenClassification.from_pretrained(
        config["model"]["path"],
        id2label=id2label,
        label2id=label2id,
    )

    accelerator = Accelerator()
    model = accelerator.prepare(
        model
    )

    unwrapped_model = accelerator.unwrap_model(model)

    # Test set evaluation
    log_msg("Test set evaluation:")
    task_evaluator = evaluate.evaluator("token-classification")
    # test_model = transformers.AutoModelForTokenClassification.from_pretrained(
    #     os.path.join(output_dir, "model")
    # )

    test_results = {}
    for (dataset_name, test_dataset) in test_datasets.items():
        test_result = task_evaluator.compute(model_or_pipeline=unwrapped_model, data=test_dataset,
                                             tokenizer=tokenizer, metric="seqeval")
        test_results[dataset_name] = test_result
        test_result_df = pd.DataFrame(test_result).loc["number"]
        log_msg("{}:\n{}\n".format(config["datasets"][dataset_name]["name"],
                                   test_result_df[
                                       ["overall_f1", "overall_accuracy", "overall_precision", "overall_recall"]]))

    end_time = time.monotonic()
    log_msg("Elapsed script time: {}\n".format(datetime.timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()
