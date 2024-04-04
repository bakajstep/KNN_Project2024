import argparse
import datetime
import glob
import logging
import os
import random
import zipfile
from warnings import simplefilter

import numpy as np
import torch
from accelerate import Accelerator
from conllu import parse
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler
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


def get_device():
    if torch.cuda.is_available():
        train_device = torch.device("cuda")
        log_msg(f"Number of GPU available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log_msg(f"Available GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        log_msg('No GPU available, using the CPU instead.')
        train_device = torch.device("cpu")

    return train_device


def conllu_to_string(conllu):
    words = [token['form'] for token in conllu]

    sentence_str = ' '.join(words)

    return sentence_str


def get_labels(conllu_sentences):
    all_labels = []

    for sentence in conllu_sentences:
        for token in sentence:
            if 'xpos' in token:
                all_labels.append([token['xpos']])

    return all_labels


def get_unique_labels(conllu_sentences):
    uniq_labels = set()

    for sentence in conllu_sentences:
        for token in sentence:
            if 'xpos' in token:
                uniq_labels.add(token['xpos'])

    return uniq_labels


def get_labels_map(uniq_labels):
    label_map = {}

    for (i, label) in enumerate(uniq_labels):
        label_map[label] = i

    return label_map


def get_attention_mask(conllu_sentences, tokenizer, max_length):
    simplefilter(action='ignore', category=FutureWarning)

    in_ids = []
    att_mask = []

    for sent in conllu_sentences:
        sent_str = ' '.join(conllu_to_string(sent))
        encoded_dict = tokenizer.encode_plus(
            sent_str,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            # max_length, # RuntimeError: The expanded size of the tensor (527) must match
            # the existing size (512) at non-singleton dimension 1.
            # Target sizes: [32, 527].  Tensor sizes: [1, 512]
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        in_ids.append(encoded_dict['input_ids'][0])

        # And its attention mask
        att_mask.append(encoded_dict['attention_mask'][0])

    return att_mask, in_ids


def get_new_labels(in_ids, lbls, lbll_map, tokenizer):
    new_lbls = []

    null_label_id = -100

    # Convert tensor IDs to tokens using BertTokenizerFast
    tokens_dict = {}
    for tensor in in_ids:
        tokens_dict.update({token_id: token for token_id, token in
                            zip(tensor.tolist(), tokenizer.convert_ids_to_tokens(tensor.tolist()))})

    for (sen, orig_labels) in zip(in_ids, lbls):
        padded_labels = []
        orig_labels_i = 0

        for token_id in sen:
            token_id = token_id.numpy().item()

            if (token_id == tokenizer.pad_token_id) or \
                    (token_id == tokenizer.cls_token_id) or \
                    (token_id == tokenizer.sep_token_id) or \
                    (tokens_dict[token_id][0:2] == '##'):

                padded_labels.append(null_label_id)
            else:
                if orig_labels_i < len(orig_labels):
                    label_str = orig_labels[orig_labels_i]
                    padded_labels.append(lbll_map[label_str])
                    orig_labels_i += 1
                else:
                    padded_labels.append(null_label_id)

        assert (len(sen) == len(padded_labels)), "sen and padded samples sizes are not same"

        new_lbls.append(padded_labels)

    return new_lbls


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


def dataset_from_sentences(sentences, tokenizer, maximum_token_length):
    labels = get_labels(sentences)
    unique_labels = get_unique_labels(sentences)
    label_map = get_labels_map(unique_labels)
    attention_masks, input_ids = get_attention_mask(sentences,
                                                    tokenizer,
                                                    maximum_token_length + 1)
    new_labels = get_new_labels(input_ids, labels, label_map, tokenizer)

    pt_input_ids = torch.stack(input_ids, dim=0)
    pt_attention_masks = torch.stack(attention_masks, dim=0)
    pt_labels = torch.tensor(new_labels, dtype=torch.long)

    dataset = TensorDataset(pt_input_ids, pt_attention_masks, pt_labels)

    return dataset


def main():
    model_dir = "../results/model"
    output_dir = "../results"
    datasets_dir = "../results/datasets"

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

    device = get_device()

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
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    sentences = parse(file_content)
                    sentences_list.extend(sentences)

        remove_files_by_extension(output_dir, '.conll')
        print(f"medival: sentences_train: {len(sentences_train)},"
              f" sentences_test: {len(sentences_test)},"
              f" sentences_validate: {len(sentences_validate)}")

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"])

    log_msg(conllu_to_string(sentences[0]))
    token_length = [len(tokenizer.encode(' '.join(conllu_to_string(i)), add_special_tokens=True))
                    for i in sentences]

    maximum_token_length = max(token_length)

    log_msg("Token lengths")
    log_msg(f'Minimum  length: {min(token_length):,} tokens')
    log_msg(f'Maximum length: {max(token_length):,} tokens')
    log_msg(f'Median length: {int(np.median(token_length)):,} tokens')

    print(f"sentences: {sentences}")

    labels = get_labels(sentences)
    unique_labels = get_unique_labels(sentences)
    label_map = get_labels_map(unique_labels)
    attention_masks, input_ids = get_attention_mask(sentences,
                                                    tokenizer,
                                                    maximum_token_length + 1)
    
    # print(f"input_ids: {input_ids}")
    print(f"labels: {labels}")
    print(f"label_map: {label_map}")

    new_labels = get_new_labels(input_ids, labels, label_map, tokenizer)

    print(f"new_labels: {new_labels}")

    pt_input_ids = torch.stack(input_ids, dim=0)

    train_dataset = dataset_from_sentences(sentences_train, tokenizer, maximum_token_length)
    val_dataset = dataset_from_sentences(sentences_validate, tokenizer, maximum_token_length)

    log_msg(f'{len(train_dataset):>5,} training samples')
    log_msg(f'{len(val_dataset):>5,} validation samples')

    batch_size = int(config["training"]["batch_size"])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                       batch_size=batch_size)

    test_pt_input_ids = torch.stack(input_ids, dim=0)
    test_pt_attention_masks = torch.stack(attention_masks, dim=0)
    test_pt_labels = torch.tensor(new_labels, dtype=torch.long)

    batch_size = 32

    print(f"{test_pt_input_ids.size()}")
    print(f"{test_pt_attention_masks.size()}")
    print(f"{test_pt_labels.size()}")

    test_prediction_data = TensorDataset(test_pt_input_ids, test_pt_attention_masks, test_pt_labels)
    test_prediction_sampler = SequentialSampler(test_prediction_data)
    test_prediction_dataloader = DataLoader(test_prediction_data, sampler=test_prediction_sampler,
                                            batch_size=batch_size)

    # Model.
    model = AutoModelForTokenClassification.from_pretrained(config["model"]["path"],
                                                            num_labels=len(label_map) + 1,
                                                            output_attentions=False,
                                                            output_hidden_states=False)
    model.cuda()

    # Load the AdamW optimizer
    config_optimizer = config["training"]["optimizer"]
    optimizer = AdamW(model.parameters(),
                      lr=float(config_optimizer["learning_rate"]),
                      betas=(float(config_optimizer["beta1"]), float(config_optimizer["beta2"])),
                      eps=float(config_optimizer["eps"]),
                      weight_decay=float(config_optimizer["weight_decay"]),
                      )

    # Number of training epochs
    epochs = int(config["training"]["num_train_epochs"])

    # Total number of training steps is number of batches * number of epochs.
    num_training_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    config_scheduler = config["training"]["lr_scheduler"]
    scheduler = get_scheduler(
        config_scheduler["name"],
        optimizer=optimizer,
        num_warmup_steps=int(config_scheduler["num_warmup_steps"]) * num_training_steps,
        num_training_steps=num_training_steps
    )

    accelerator = Accelerator()

    (model, optimizer, train_dataloader, validation_dataloader, test_prediction_dataloader,
     scheduler) = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, test_prediction_dataloader,
        scheduler)

    # Setting the random seed for reproducibility, etc.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []

    #################
    # Training loop #
    #################
    for epoch_i in range(0, epochs):
        ############
        # Training #
        ############
        log_msg(f'======== Epoch {epoch_i + 1} / {epochs} ========')
        log_msg('Training...')

        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                # Report progress.
                log_msg(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.')

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs[0]

            total_loss += loss.item()

            # loss.backward()
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        log_msg(f"  Average training loss: {avg_train_loss:.2f}")

        val_predictions, val_true_labels = [], []

        # TODO jestli pouzit accelerator i v evaluations
        ##############
        # Evaluation #
        ##############
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and

            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            val_predictions.append(logits)
            val_true_labels.append(label_ids)

        # First, combine the results across the batches.
        all_val_predictions = np.concatenate(val_predictions, axis=0)
        all_val_true_labels = np.concatenate(val_true_labels, axis=0)

        log_msg("After flattening the batches, the predictions have shape:")
        log_msg(f"    {all_val_predictions.shape}")

        # Next, let's remove the third dimension (axis 2), which has the scores
        # for all 18 labels.

        # For each token, pick the label with the highest score.
        val_predicted_label_ids = np.argmax(all_val_predictions, axis=2)

        log_msg("\nAfter choosing the highest scoring label for each token:")
        log_msg(f"    {val_predicted_label_ids.shape}")

        # Eliminate axis 0, which corresponds to the sentences.
        val_predicted_label_ids = np.concatenate(val_predicted_label_ids, axis=0)
        all_val_true_labels = np.concatenate(all_val_true_labels, axis=0)

        log_msg("\nAfter flattening the sentences, we have predictions:")
        log_msg(f"    {val_predicted_label_ids.shape}")
        log_msg("and ground truth:")
        log_msg(f"    {all_val_true_labels.shape}")

        val_token_predictions = []
        val_token_labels = []

        # For each of the input tokens in the dataset...
        for i, label in enumerate(all_val_true_labels):

            # If it's not a token with a null label...
            if label != -100:
                # Add the prediction and the ground truth to their lists.
                val_token_predictions.append(val_predicted_label_ids[i])
                val_token_labels.append(label)

        val_f1 = f1_score(val_token_labels, val_token_predictions, average='micro')

        log_msg(f"F1 score: {val_f1:.2%}")

        ################
        # Saving model #
        ################
        # https://huggingface.co/course/en/chapter7/2?fw=pt
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(model_dir)

    # Testing
    log_msg(f'Predicting labels for {len(pt_input_ids):,} test sentences...')

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # TODO jestli pouzit accelerator i v testovani
    # https://github.com/roman-janik/diploma_thesis_program/blob/main/train_ner_model.py#L259
    # Predict
    for batch in test_prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    log_msg('    DONE.')

    # First, combine the results across the batches.
    all_predictions = np.concatenate(predictions, axis=0)
    all_true_labels = np.concatenate(true_labels, axis=0)

    log_msg("After flattening the batches, the predictions have shape:")
    log_msg(f"    {all_predictions.shape}")

    # Next, let's remove the third dimension (axis 2), which has the scores
    # for all 18 labels.

    # For each token, pick the label with the highest score.
    predicted_label_ids = np.argmax(all_predictions, axis=2)

    log_msg("\nAfter choosing the highest scoring label for each token:")
    log_msg(f"    {predicted_label_ids.shape}")

    # Eliminate axis 0, which corresponds to the sentences.
    predicted_label_ids = np.concatenate(predicted_label_ids, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    log_msg("\nAfter flattening the sentences, we have predictions:")
    log_msg(f"    {predicted_label_ids.shape}")
    log_msg("and ground truth:")
    log_msg(f"    {all_true_labels.shape}")

    real_token_predictions = []
    real_token_labels = []

    # For each of the input tokens in the dataset...
    for i, label in enumerate(all_true_labels):

        # If it's not a token with a null label...
        if label != -100:
            # Add the prediction and the ground truth to their lists.
            real_token_predictions.append(predicted_label_ids[i])
            real_token_labels.append(label)

    f1 = f1_score(real_token_labels, real_token_predictions, average='micro')

    log_msg(f"F1 score: {f1:.2%}")


if __name__ == "__main__":
    main()
