import os
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """\
This is an open dataset of sentences from 19th and 20th century letterpress reprints of documents 
from the Hussite era. The dataset contains a corpus for language modeling and human annotations for 
named entity recognition (NER).
"""

_TRAINING_FILE = "dataset_ner_fuzzy-regex_all_all_training.conll"
_DEV_FILE = "dataset_ner_fuzzy-regex_all_all_validation.conll"
_TEST_FILE = "dataset_ner_fuzzy-regex_all_all_testing.conll"


class MedivalConllConfig(datasets.BuilderConfig):
    """BuilderConfig for Medival"""

    def __init__(self, **kwargs):
        """BuilderConfig for Medival.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedivalConllConfig, self).__init__(**kwargs)


class MedivalConll(datasets.GeneratorBasedBuilder):
    """Medival dataset."""

    BUILDER_CONFIGS = [
        MedivalConllConfig(name="Medival", version=datasets.Version("2.0.0"),
                           description="Historic dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-LOC",
                                "I-LOC"
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5024",
        )

    def _split_generators(self, dl_manager, dataset_path="../parsers/custom/medival"):
        """Returns SplitGenerators."""
        data_files = {
            "train": os.path.join(dataset_path, _TRAINING_FILE),
            "dev": os.path.join(dataset_path, _DEV_FILE),
            "test": os.path.join(dataset_path, _TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # Medival tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[9].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
