# KNN_Project2024

This repository contains a source code for the paper "Automated Data Labeling for the Czech Named Entity Recognition".

## Repository structure

|   |   |
|---|---|
| *configs/* | configuration YAML files for training |
| *metacentrum_scripts/* | scripts for training and evaluation on the MetaCentrum nodes |
| *parsers/* | datasets parsers |
| *README.md* | this README |
| *ner_trainer.py* | script for NER training with the specific configuration setting |
| *requirements.txt* | required libraries for running |
| *text_classification.py* | script for running the Named Entity Recognition |
| *tmp_list_of_models_and_datasets.txt* | temporary file containing a list of found models |

## Trained models
The fine-tuned models can be downloaded from the [public storage](https://drive.google.com/drive/folders/1-4beN42ym1WDLJqviYmY0mOoxOyjH3ey?usp=sharing).

## Results
The following table captures the performances of the used models.

| Model                    | Configuration                              | F1 score --- own (%) | F1 score --- target (%) |
| ------------------------ | ------------------------------------------ | -------------------- | ----------------------- |
| BERT Fine-Tuned, Custom  | [YAML](configs/cnec_lr_5e5_12_epochs.yaml) | 75.48                | 71.04                   |

Note:
- **F1 score --- own** --- the performance on the testing subset of the data used for training
- **F1 score --- target** --- the performance on the manually labelled subset of the target unlabeled data

## Usage

### 1. Manually
#### Install
Tested with Python 3.9.
```bash
pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt
```
#### Train NER model
```bash
python ner_trainer.py --config <config.yaml>
```
#### Classify
```bash
python text_classification.py --model <model_dir_path/>
```

### 2. Automatically
The running scripts are prepared for the MetaCentrum nodes.

#### Train NER model
```bash
./train.sh <branch_name> <config_file_name> <timeout>
```

#### Classify
```bash
./test.sh <branch_name> <model_dir_path/> <timeout>
```
