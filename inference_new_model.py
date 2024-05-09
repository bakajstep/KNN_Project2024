import argparse
from transformers import pipeline
from yaml import safe_load


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    args = parser.parse_args()
    return args


def convert_to_conllu(text, entities):
    words = text.split()
    tagged_words = [(word, 'O') for word in words]

    for entity in entities:
        start_index = entity['start']
        end_index = entity['end']
        entity_type = entity['entity_group']

        for i, (word, tag) in enumerate(tagged_words):
            word_index = text.index(word)
            if word_index >= start_index and word_index < end_index:
                prefix = 'B-' if word_index == start_index else 'I-'
                tagged_words[i] = (word, prefix + entity_type)

    result = '\n'.join([f"{word} {tag}" for word, tag in tagged_words])
    return result


def main():
    args = parse_arguments()

    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)

    model_checkpoint = config["model"]["path"]
    token_classifier = pipeline(
        "token-classification", model=model_checkpoint, aggregation_strategy="simple"
    )

    sentences = ["Jmenuji se Sylvain a pracuji ve společnosti Hugging Face v Brooklynu."]

    # Inicializace proměnné pro výstup
    full_output = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # Přeskočení prázdných řádků
            classification_result = token_classifier(sentence)
            entities = [
                {'entity_group': entity['entity_group'], 'start': entity['start'], 'end': entity['end']}
                for entity in classification_result
            ]

            conllu_output = convert_to_conllu(sentence, entities)
            full_output.append(conllu_output)
            full_output.append('')  # Přidání prázdného řádku mezi větami

    # Výpis celkového výstupu
    print('\n'.join(full_output))


if __name__ == "__main__":
    main()
