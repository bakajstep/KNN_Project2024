import random
from warnings import simplefilter

import numpy as np
from conllu import parse

from sklearn.metrics import f1_score

import argparse
from yaml import safe_load

from accelerate import Accelerator

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler # get_linear_schedule_with_warmup
import torch
import logging
import os
import datetime


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
        log_msg("Number of GPU available: {}".format(torch.cuda.device_count()))
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
            max_length= 512, # max_length, # RuntimeError: The expanded size of the tensor (527) must match the existing size (512) at non-singleton dimension 1.  Target sizes: [32, 527].  Tensor sizes: [1, 512]
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
        tokens_dict.update({token_id: token for token_id, token in zip(tensor.tolist(), tokenizer.convert_ids_to_tokens(tensor.tolist()))})

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
    log_msg("{:<24}{}\n{:<24}{}".format(
        "Name:", exp_name.removeprefix("exp_configs_ner/").removesuffix(".yaml"), "Description:", config["desc"]))
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n".format(
        "Start time:", ct, "Model:", config["model"]["name"],
        "Datasets:", [dts["name"] for dts in config["datasets"].values()]))

    cf_t = config["training"]
    log_msg("Parameters:\n{:<24}{}\n{:<24}{}".format(
        "Num train epochs:", cf_t["num_train_epochs"], "Batch size:", cf_t["batch_size"]))
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n{:<24}{}".format(
        "Learning rate:", cf_t["optimizer"]["learning_rate"], "Weight decay:", cf_t["optimizer"]["weight_decay"],
        "Lr scheduler:",
        cf_t["lr_scheduler"]["name"], "Warmup steps:", cf_t["lr_scheduler"]["num_warmup_steps"]))
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}".format(
        "Beta1:", cf_t["optimizer"]["beta1"], "Beta2:", cf_t["optimizer"]["beta2"], "Epsilon:",
        cf_t["optimizer"]["eps"]))


def main():
    model_dir = "../results/model"
    output_dir = "../results"

    args = parse_arguments()

    # Load a config file.
    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)

    # Start logging, print experiment configuration
    logging.basicConfig(filename=os.path.join(output_dir, "experiment_results.txt"), level=logging.INFO,
                        encoding='utf-8', format='%(message)s')
    log_msg("Experiment summary:\n")
    log_summary(args.config, config)
    log_msg("-" * 80 + "\n")

    device = get_device()

    sentences = []
    if "cnec2" in config["datasets"]:
        dataset_dir = "ourdatasets/cnec2_extended"

        dataset_files = {
            "train.conll": "",
            "test.conll": "",
            "dev.conll": "",
        }

        for key in dataset_files.keys():
            file_path = os.path.join(dataset_dir, key)
            with open(file_path, 'r', encoding='utf-8') as file:
                dataset_files[key] += file.read()
        sentences.extend(parse(dataset_files["train.conll"]))
    if "wikiann" in config["datasets"]:
        dataset_dir = "ourdatasets/wikiann"

        dataset_files = {
            "train.conll": "",
            "test.conll": "",
            "dev.conll": "",
        }

        for key in dataset_files.keys():
            file_path = os.path.join(dataset_dir, key)
            with open(file_path, 'r', encoding='utf-8') as file:
                dataset_files[key] += file.read()
        sentences.extend(parse(dataset_files["train.conll"]))

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"])
    
    log_msg(conllu_to_string(sentences[0]))
    TokenLength = [len(tokenizer.encode(' '.join(conllu_to_string(i)), add_special_tokens=True)) for i in sentences]

    maximum_token_length = max(TokenLength)

    log_msg("Token lengths")
    log_msg('Minimum  length: {:,} tokens'.format(min(TokenLength)))
    log_msg('Maximum length: {:,} tokens'.format(max(TokenLength)))
    log_msg('Median length: {:,} tokens'.format(int(np.median(TokenLength))))

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

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(pt_input_ids, pt_attention_masks, pt_labels)

    # Create a 90-10 train-validation split.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    log_msg('{:>5,} training samples'.format(train_size))
    log_msg('{:>5,} validation samples'.format(val_size))

    batch_size = int(config["training"]["batch_size"])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)    

    # Test data.
    test_sentences = parse(dataset_files["test.conll"])
    test_labels = get_labels(test_sentences)
    test_unique_labels = get_unique_labels(test_sentences)
    test_label_map = get_labels_map(test_unique_labels)
    # TODO is it needed? because it is unused
    test_attention_masks, test_input_ids = get_attention_mask(test_sentences,
                                                              tokenizer,
                                                              maximum_token_length + 1)
    test_new_labels = get_new_labels(test_input_ids, test_labels, test_label_map, tokenizer)

    test_pt_input_ids = torch.stack(input_ids, dim=0)
    test_pt_attention_masks = torch.stack(attention_masks, dim=0)
    test_pt_labels = torch.tensor(new_labels, dtype=torch.long)

    batch_size = 32

    test_prediction_data = TensorDataset(test_pt_input_ids, test_pt_attention_masks, test_pt_labels)
    test_prediction_sampler = SequentialSampler(test_prediction_data)
    test_prediction_dataloader = DataLoader(test_prediction_data, sampler=test_prediction_sampler, batch_size=batch_size)

    # Model.
    model = AutoModelForTokenClassification.from_pretrained(config["model"]["path"], num_labels=len(label_map) + 1,
                                                            output_attentions=False, output_hidden_states=False)
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

    model, optimizer, train_dataloader, validation_dataloader, test_prediction_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, test_prediction_dataloader, scheduler
    )

    # Setting the random seed for reproducibility, etc.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []

    # TODO evaluace behem trenovani jednotlivych epoch?
    # https://huggingface.co/learn/nlp-course/en/chapter7/2?fw=pt
    #################
    # Training loop #
    #################
    for epoch_i in range(0, epochs):
        ############
        # Training #
        ############
        log_msg('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        log_msg('Training...')

        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                # Report progress.
                log_msg('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]

            total_loss += loss.item()

            # loss.backward()
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        log_msg("  Average training loss: {0:.2f}".format(avg_train_loss))

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
        for i in range(len(all_val_true_labels)):

            # If it's not a token with a null label...
            if not all_val_true_labels[i] == -100:
                # Add the prediction and the ground truth to their lists.
                val_token_predictions.append(val_predicted_label_ids[i])
                val_token_labels.append(all_val_true_labels[i])

        val_f1 = f1_score(val_token_labels, val_token_predictions, average='micro')

        log_msg("F1 score: {:.2%}".format(val_f1))

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
    log_msg('Predicting labels for {:,} test sentences...'.format(len(pt_input_ids)))

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
    for i in range(len(all_true_labels)):

        # If it's not a token with a null label...
        if not all_true_labels[i] == -100:
            # Add the prediction and the ground truth to their lists.
            real_token_predictions.append(predicted_label_ids[i])
            real_token_labels.append(all_true_labels[i])

    f1 = f1_score(real_token_labels, real_token_predictions, average='micro')

    log_msg("F1 score: {:.2%}".format(f1))

if __name__ == "__main__":
    main()
