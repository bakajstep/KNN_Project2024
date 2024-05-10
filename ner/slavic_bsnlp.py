import os
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """
"""

_DESCRIPTION = """
"""

_TRAINING_FILE = "train.conll"
_TEST_FILE = "test.conll"
_VALIDATION_FILE = "validation.conll"


class Cnec2_0ConllConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        """BuilderConfig for Slavic.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Cnec2_0ConllConfig, self).__init__(**kwargs)


class Cnec2_0Conll(datasets.GeneratorBasedBuilder):
    """slavic dataset."""

    BUILDER_CONFIGS = [
        Cnec2_0ConllConfig(name="slavic_bsnlp", version=datasets.Version("2.0.0"),
                           description="slavic_bsnlp dataset"),
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
                            names=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
                                   "B-PRO", "I-PRO", "B-EVT", "I-EVT"]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="example.com",
            citation=_CITATION,
        )

    # def _split_generators(self, dl_manager, dataset_path="../../../our_datasets/cnec2.0_extended"):
    def _split_generators(self, dl_manager, dataset_path="our_datasets/slavic"):
        """Returns SplitGenerators."""
        data_files = {
            "train": os.path.join(dataset_path, _TRAINING_FILE),
            "test": os.path.join(dataset_path, _TEST_FILE),
            "validation": os.path.join(dataset_path, _VALIDATION_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": data_files["test"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": data_files["validation"]}),
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
                    # CNEC 2.0 CoNNL tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[1])
                    ner_tags.append(splits[2].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
