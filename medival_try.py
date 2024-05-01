import argparse
import datetime
import glob
import logging
import os
import gc
import zipfile
from conllu import parse
from yaml import safe_load

from parsers.cnec2_extended.cnec2_extended import get_cnec2_extended
from parsers.slavic.slavic_bsnlp import prepare_slavic
from parsers.util import remove_files_by_extension
from parsers.wikiann_cs.wikiann_cs import prepare_wikiann
from parsers.medival.medival_parser import prepare_medival


# https://github.com/roman-janik/diploma_thesis_program/blob/a23bfaa34d32f92cd17dc8b087ad97e9f5f0f3e6/train_ner_model.py#L28
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    # parser.add_argument('--results_csv', required=True, help='Results CSV file.')
    args = parser.parse_args()
    return args


def conllu_to_string(conllu):
    words = [token['form'] for token in conllu]

    sentence_str = ' '.join(words)

    return sentence_str


def log_msg(msg: str):
    print(msg)
    logging.info(msg)


def log_summary(exp_name: str, config: dict):
    log_msg(
        f"{'Name:':<24}{exp_name.removeprefix('exp_configs_ner/').removesuffix('.yaml')}\n"
        f"{'Description:':<24}{config['desc']}")
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg(
        f"{'Start time:':<24}{ct}\n{'Model:':<24}{config['model']['name']}\n"
        f"{'Datasets:':<24}{[dts['name'] for dts in config['datasets'].values()]}\n")

    cf_t = config["training"]
    log_msg(
        f"Parameters:\n"
        f"{'Num train epochs:':<24}{cf_t['num_train_epochs']}\n"
        f"{'Batch size:':<24}{cf_t['batch_size']}")
    log_msg(
        f"{'Learning rate:':<24}{cf_t['optimizer']['learning_rate']}\n"
        f"{'Weight decay:':<24}{cf_t['optimizer']['weight_decay']}\n"
        f"{'Lr scheduler:':<24}{cf_t['lr_scheduler']['name']}\n"
        f"{'Warmup steps:':<24}{cf_t['lr_scheduler']['num_warmup_steps']}")
    log_msg(
        f"{'Beta1:':<24}{cf_t['optimizer']['beta1']}\n"
        f"{'Beta2:':<24}{cf_t['optimizer']['beta2']}\n"
        f"{'Epsilon:':<24}{cf_t['optimizer']['eps']}")


def main():
    output_dir = "./results"
    datasets_dir = "./results/datasets"

    args = parse_arguments()

    # Load a config file.
    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = safe_load(config_file)

    # Start logging, print experiment configuration
    logging.basicConfig(filename=os.path.join(output_dir, "experiment_results.txt"),
                        level=logging.INFO,
                        encoding='utf-8', format='%(message)s')
    log_msg("Experiment summary:\n")
    log_summary(args.config, config)
    log_msg("-" * 80 + "\n")

    sentences_train = []
    sentences_test = []
    sentences_validate = []
    if "cnec2" in config["datasets"]:
        log_msg("Using cnec2 dataset")
        if not os.path.exists(f"{datasets_dir}/cnec2.zip"):
            log_msg("Downloading cnec2 dataset")
            get_cnec2_extended(config["datasets"]["cnec2"]["url_path"], datasets_dir, "cnec2")

        with zipfile.ZipFile(f"{datasets_dir}/cnec2.zip", 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)

        dataset_info = [
            ("train.conll", sentences_train),
            ("test.conll", sentences_test),
            ("dev.conll", sentences_validate)
        ]

        for filename, sentences_list in dataset_info:
            file_path = os.path.join(datasets_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                sentences = parse(file_content)
                sentences_list.extend(sentences)

        remove_files_by_extension(output_dir, '.conll')
        print(f"Cnec2: sentences_train: {len(sentences_train)},"
              f" sentences_test: {len(sentences_test)},"
              f" sentences_validate: {len(sentences_validate)}")

    if "wikiann" in config["datasets"]:
        log_msg("Using wikiann dataset")
        if not os.path.exists(f"{datasets_dir}/wikiann.zip"):
            log_msg("Downloading wikiann dataset")
            prepare_wikiann(datasets_dir, "wikiann")

        with zipfile.ZipFile(f"{datasets_dir}/wikiann.zip", 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)

        dataset_info = [
            ("train.conll", sentences_train),
            ("test.conll", sentences_test),
            ("validation.conll", sentences_validate)
        ]

        for filename, sentences_list in dataset_info:
            file_path = os.path.join(datasets_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                sentences = parse(file_content)
                sentences_list.extend(sentences)

        remove_files_by_extension(output_dir, '.conll')
        print(f"Wikiann: sentences_train: {len(sentences_train)},"
              f" sentences_test: {len(sentences_test)},"
              f" sentences_validate: {len(sentences_validate)}")

    if "slavic" in config["datasets"]:
        log_msg("Using slavic dataset")
        if not os.path.exists(f"{datasets_dir}/slavic.zip"):
            log_msg("Downloading slavic dataset")
            prepare_slavic(config["datasets"]["slavic"]["url_train"],
                           config["datasets"]["slavic"]["url_test"],
                           datasets_dir, "slavic")

        with zipfile.ZipFile(f"{datasets_dir}/slavic.zip", 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)

        dataset_info = [
            ("train.conll", sentences_train),
            ("test.conll", sentences_test),
        ]

        # Načtení a zpracování každého datasetu
        for filename, sentences_list in dataset_info:
            file_path = os.path.join(datasets_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                sentences = parse(file_content)
                sentences_list.extend(sentences)

        remove_files_by_extension(output_dir, '.conll')
        print(f"Slavic: sentences_train: {len(sentences_train)},"
              f" sentences_test: {len(sentences_test)},"
              f" sentences_validate: {len(sentences_validate)}")

    if "medival" in config["datasets"]:
        log_msg("Using medival dataset")
        if not os.path.exists(f"{datasets_dir}/medival.zip"):
            log_msg("Downloading medival dataset")
            prepare_medival(config["datasets"]["medival"]["url_path"], datasets_dir, "medival")

        log_msg("Medival processed")
        with zipfile.ZipFile(f"{datasets_dir}/medival.zip", 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)

        patterns = {
            "*training*.conll": sentences_train,
            "*test*.conll": sentences_test,
            "*validation*.conll": sentences_validate
        }

        for pattern, sentences_list in patterns.items():
            # Vytvoření plného vzoru cesty s použitím glob
            full_pattern = os.path.join(datasets_dir, pattern)
            for file_path in glob.glob(full_pattern):
                log_msg(file_path)
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Čtení souboru po blocích oddělených prázdnými řádky
                    buffer = []
                    for line in file:
                        if line.strip():  # Pokud řádek není prázdný, přidáme ho do bufferu
                            buffer.append(line)
                        else:  # Když narazíme na prázdný řádek, zpracujeme nahromaděný buffer
                            if buffer:
                                sentence = parse("".join(buffer))[0]
                                sentences_list.append(sentence)
                                buffer = []  # Reset bufferu po zpracování
                    if buffer:  # Zpracování zbývajícího bufferu po skončení souboru
                        sentence = parse("".join(buffer))[0]
                        sentences_list.append(sentence)

        remove_files_by_extension(output_dir, '.conll')
        print(f"medival: sentences_train: {len(sentences_train)},"
              f" sentences_test: {len(sentences_test)},"
              f" sentences_validate: {len(sentences_validate)}")


if __name__ == "__main__":
    main()
