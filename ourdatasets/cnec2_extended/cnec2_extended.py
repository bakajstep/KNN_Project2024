import zipfile
from io import BytesIO
import requests


def get_dataset(url_path):
    response = requests.get(url_path)
    zip_file = zipfile.ZipFile(BytesIO(response.content))

    root_dir = 'cnec2.0_extended/'
    files_to_extract = ['dev.conll', 'train.conll', 'test.conll']

    # Mapování pouze specifické části anotace
    annotation_mapping = {
        "P": "PER",
        "G": "LOC",
        "I": "ORG"
    }

    # Vytvoření výstupní složky, pokud neexistuje
    output_dir = 'ourdatasets/cnec2_extended'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in files_to_extract:
        with zip_file.open(root_dir + file_name) as file:
            lines = file.read().decode('utf-8').splitlines()
            processed_lines = []
            line_number = 1
            for line in lines:
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) >= 4:
                        word, lemma, morph, annotation = parts[:4]
                        annotation_parts = annotation.split("-")
                        if len(annotation_parts) == 2:
                            prefix, label = annotation_parts
                            mapped_label = annotation_mapping.get(label, label)
                            mapped_annotation = f"{prefix}-{mapped_label}"
                        else:
                            mapped_annotation = annotation
                        processed_lines.append("\t".join([str(line_number), word, lemma, morph, mapped_annotation]))
                    else:
                        processed_lines.append("\t".join([str(line_number)] + parts))
                else:
                    processed_lines.append("")
                line_number += 1

            # Zápis zpracovaného obsahu do souboru
            output_file_path = os.path.join(output_dir, file_name)
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write("\n".join(processed_lines))

    print(f"Files have been processed and saved to {output_dir}")


def get_cnec2_extended():
    get_dataset("https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3493/cnec2.0_extended.zip")
