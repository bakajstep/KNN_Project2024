# KNN_Project2024

This repository contains a source code for the paper "Automated Data Labeling for the Czech Named Entity Recognition".

## Repository structure

|   |   |
|---|---|
| *configs/* | configuration YAML files for training |
| *metacentrum_scripts/* | scripts for training and evaluation on the MetaCentrum nodes |
| *parsers/* | datasets parsers |
| *README.md* | this README |
| *train_ner_model.py* | script for NER training with the specific configuration setting |
| *requirements.txt* | required libraries for running |
| *eval_ner_model.py* | evaluation of a model on a test dataset |
| *chain_of_trust_eval.py* | evaluation of a chain of trust system on a test dataset |
| *inference_new_model.py* | script for running the Named Entity Recognition |
| *count_tags.py* |  script for counting tags in the created dataset|

## Trained models and dataset
The fine-tuned models and created dataset can be downloaded from the [public storage](https://vutbr-my.sharepoint.com/:f:/g/personal/xchoch09_vutbr_cz/EnovG2j8eK9Fit6kT3NUuO8Bxko2-tPwLVLA7goprsSsFw?e=JJNR7N).

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

#### Evaluation
```bash
python eval_ner_model.py --model <model_dir_path/>
```

#### Inference
```bash
python inference_new_model.py --model <model_dir_path/>
```

### 2. Automatically
The running scripts are prepared for the MetaCentrum nodes.

#### Train NER model
```bash
./train.sh <branch_name> <config_file_name> <timeout>
```

#### Evaluation
```bash
./evaluation.sh <branch_name> <model_dir_path/> <timeout>
```

#### Inference
```bash
./interference_chot.sh <branch_name> <model_dir_path/> <timeout>
```
