# Author: Roman Janík
# Script for local loading CNEC 2.0 CoNNL dataset and converting it to Hugging Face dataset format.
#
# This script if a modified version of conll2003/conll2003.py script by HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@InProceedings{CNEC-2.0-CoNLL,
    author="Konkol, Michal
    and Konop{\'i}k, Miloslav",
    editor="Habernal, Ivan
    and Matou{\v{s}}ek, V{\'a}clav",
    title="CRF-Based Czech Named Entity Recognizer and Consolidation of Czech NER Research",
    booktitle="Text, Speech, and Dialogue",
    year="2013",
    publisher="Springer Berlin Heidelberg",
    address="Berlin, Heidelberg",
    pages="153--160",
    isbn="978-3-642-40585-3"
"""

_DESCRIPTION = """\
This is a Czech Named Entity Corpus 2.0 transformed into the CoNLL format. Each line in
the corpus contains information about one word/token. The first column is the actual
word, the second column is a lemma, the third column is a morphological tag, and the fourth
column is a Named Entity class in a BIO format. An empty line is a sentence separator.
The lemmas and tags are created by an automatic tagger. The structure of the tags is
described at:

  http://ufal.mff.cuni.cz/pdt2.0/doc/manuals/en/m-layer/html/ch02.html

The original corpus can be downloaded from

  http://hdl.handle.net/11858/00-097C-0000-0023-1B22-8
"""

_TRAINING_FILE = "train.conll"
_DEV_FILE = "dev.conll"
_TEST_FILE = "test.conll"


class Cnec2_0ConllConfig(datasets.BuilderConfig):
    """BuilderConfig for CNEC 2.0 CoNNL"""

    def __init__(self, **kwargs):
        """BuilderConfig for CNEC 2.0 CoNNL.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Cnec2_0ConllConfig, self).__init__(**kwargs)


class Cnec2_0Conll(datasets.GeneratorBasedBuilder):
    """CNEC 2.0 CoNNL dataset."""

    BUILDER_CONFIGS = [
        Cnec2_0ConllConfig(name="cnec2_0conll", version=datasets.Version("2.0.0"), description="CNEC 2.0 CoNNL dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "lemmas": datasets.Sequence(datasets.Value("string")),
                    "morph_tags": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-P",
                                "I-P",
                                "B-I",
                                "I-I",
                                "B-G",
                                "I-G",
                                "B-T",
                                "I-T",
                                "B-O",
                                "I-O",
                                "B-M",
                                "I-M",
                                "B-A",
                                "I-A"
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="http://hdl.handle.net/11234/1-3493",
            citation=_CITATION,
        )

    #def _split_generators(self, dl_manager, dataset_path="../../../our_datasets/cnec2.0_extended"):
    def _split_generators(self, dl_manager, dataset_path="our_datasets/cnec2.0_extended"):
        """Returns SplitGenerators."""
        data_files = {
            "train": os.path.join(dataset_path, _TRAINING_FILE),
            "dev": os.path.join(dataset_path, _DEV_FILE),
            "test": os.path.join(dataset_path, _TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            lemmas = []
            morph_tags = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "lemmas": lemmas,
                            "morph_tags": morph_tags,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        lemmas = []
                        morph_tags = []
                        ner_tags = []
                else:
                    # CNEC 2.0 CoNNL tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    lemmas.append(splits[1])
                    morph_tags.append(splits[2])
                    ner_tags.append(splits[3].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "lemmas": lemmas,
                    "morph_tags": morph_tags,
                    "ner_tags": ner_tags,
                }
