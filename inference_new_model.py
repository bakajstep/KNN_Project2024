import argparse

from transformers import pipeline
from yaml import safe_load

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    #parser.add_argument('--results_csv', required=True, help='Results CSV file.')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)

    # Replace this with your own checkpoint
    model_checkpoint = config["model"]["path"]
    token_classifier = pipeline(
        "token-classification", model=model_checkpoint, aggregation_strategy="simple"
    )
    classification_result = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    print(classification_result)

if __name__ == "__main__":
    main()
