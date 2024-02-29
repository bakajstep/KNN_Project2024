from io import BytesIO
from warnings import simplefilter

import numpy as np
from conllu import parse
import requests
import zipfile

from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer
import csv
import torch


def get_dataset():
    url = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3493/cnec2.0_extended.zip'

    response = requests.get(url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))

    root_dir = 'cnec2.0_extended/'
    files_to_extract = ['dev.conll', 'train.conll', 'test.conll']

    file_contents = {}

    for file_name in files_to_extract:
        with zip_file.open(root_dir + file_name) as file:
            lines = file.read().decode('utf-8').splitlines()
            numbered_lines = []
            line_number = 1
            for line in lines:
                if line.strip():
                    numbered_lines.append(f"{line_number}\t{line}")
                    line_number += 1
                else:
                    numbered_lines.append("")
                    # line_number = 1
            file_contents[file_name] = "\n".join(numbered_lines)

    return file_contents


def get_device():
    if torch.cuda.is_available():
        train_device = torch.device("cuda")
        print(torch.cuda.device_count())
        print('Available:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
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
                all_labels.append(token['xpos'])

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


def get_attention_mask(conllu_sentences):
    simplefilter(action='ignore', category=FutureWarning)

    in_ids = []
    att_mask = []

    for sent in conllu_sentences:
        sent_str = ' '.join(conllu_to_string(sent))
        encoded_dict = tokenizer.encode_plus(
            sent_str,
            add_special_tokens=True,
            truncation=True,
            max_length=55,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        in_ids.append(encoded_dict['input_ids'][0])

        # And its attention mask
        att_mask.append(encoded_dict['attention_mask'][0])

    return att_mask, in_ids


def get_new_labels(in_ids, lbls, lbll_map):
    new_lbls = []

    # The special label ID we'll give to "extra" tokens.
    null_label_id = -100

    print(len(in_ids), len(lbls))

    for (sen, orig_labels) in zip(in_ids, lbls):

        padded_labels = []

        orig_labels_i = 0

        for token_id in sen:

            token_id = token_id.numpy().item()

            if (token_id == tokenizer.pad_token_id) or \
                    (token_id == tokenizer.cls_token_id) or \
                    (token_id == tokenizer.sep_token_id):

                padded_labels.append(null_label_id)

            elif tokenizer.ids_to_tokens[token_id][0:2] == '##':

                padded_labels.append(null_label_id)

            else:

                label_str = orig_labels[orig_labels_i]

                padded_labels.append(lbll_map[label_str])

                orig_labels_i += 1

        assert (len(sen) == len(padded_labels))

        new_lbls.append(padded_labels)

    return new_lbls


if __name__ == '__main__':
    device = get_device()

    dataset_files = get_dataset()
    sentences = parse(dataset_files["train.conll"])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # TokenLength = [len(tokenizer.encode(' '.join(conllu_to_string(i)), add_special_tokens=True)) for i in sentences]
    # print('Minimum  length: {:,} tokens'.format(min(TokenLength)))
    # print('Maximum length: {:,} tokens'.format(max(TokenLength)))
    # print('Median length: {:,} tokens'.format(int(np.median(TokenLength))))

    labels = get_labels(sentences)
    unique_labels = get_unique_labels(sentences)
    label_map = get_labels_map(unique_labels)
    attention_masks, input_ids = get_attention_mask(sentences)
    new_labels = get_new_labels(input_ids, labels, label_map)

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

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
