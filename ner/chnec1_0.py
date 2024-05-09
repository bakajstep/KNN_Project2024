# Author: Roman Janík
# Script for local loading CHNEC 1.0 datasets and converting it to Hugging Face dataset format.
#
# This script if a modified version of conll2003/conll2003.py script by HuggingFace Datasets Authors.
#
# NOTICE: There is a wrong annotation on line 16853 in train split (CHNEC_v0.1_train.conll):
# "Biůrger	DE	_	I-p"
# - lemma (2.) and language (3.) columns are swapped! This will cause an error during loading,
# needs to be corrected before running loading script!
#
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
@inproceedings{CHNEC,
    title = "{C}zech Historical Named Entity Corpus v 1.0",
    author = "Hubkov{\'a}, Helena  and
    Kral, Pavel  and
    Pettersson, Eva",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.549",
    pages = "4458--4465",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
"""

_DESCRIPTION = """\
Czech Historical Named Entity Corpus v 1.0 is a collection of annotated texts for historical Czech named-entity
recognition. It is composed of Czech texts from newspaper "Posel od Čerchova" from the second half of 19th century. 
We specified following the basic NE-types: Personal names, Institutions, Geographical names, Time expressions
and Artifact names / Objects. More in annotation_manual.pdf.
Every  token  is  placed  in  a  separate  line  which contains four columns, each column separated by a space.
The first column is the token, whereas the second one is reserved for lemma (non-specified in our case,
represented by an underscore symbol). The third column contains information about the language. Most tokens
are Czech ones (”CZ”),  but we can also find some tokens in German (”DE”), French ("FR") or Latin (”LA”).
The last column is used to describe the named entity type.
We also used "IOB" notations to indicate the first word in a multiword entity (tag "B" as "beginning"),
and inside words for all other NE units (tag "I" as "internal"). All tokens that are not a named entity
are tagged as "O" - "outside". 
Each sentence is separated by empty line.

This corpus is available only for research purposes for free. Commercial use in any form is strictly excluded. 
"""

_TRAINING_FILE = "CHNEC_v0.1_train.conll"
_DEV_FILE = "CHNEC_v0.1_dev.conll"
_TEST_FILE = "CHNEC_v0.1_test.conll"


class Chnec1_0Config(datasets.BuilderConfig):
    """BuilderConfig for CHNEC 1.0"""

    def __init__(self, **kwargs):
        """BuilderConfig for CHNEC 1.0.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Chnec1_0Config, self).__init__(**kwargs)


class Chnec1_0(datasets.GeneratorBasedBuilder):
    """CHNEC 1.0 dataset."""

    BUILDER_CONFIGS = [
        Chnec1_0Config(name="chnec1_0", version=datasets.Version("1.0.0"), description="CHNEC 1.0 dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "lemmas": datasets.Sequence(datasets.Value("string")),
                    "language": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "CZ",
                                "DE",
                                "FR",
                                "LA"
                            ]
                        )
                    ),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-p",
                                "I-p",
                                "B-i",
                                "I-i",
                                "B-g",
                                "I-g",
                                "B-t",
                                "I-t",
                                "B-o",
                                "I-o"
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="http://chnec.kiv.zcu.cz/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager, dataset_path="our_datasets/chnec1.0"):
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
            language = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "lemmas": lemmas,
                            "language": language,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        lemmas = []
                        language = []
                        ner_tags = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    lemmas.append(splits[1])
                    language.append(splits[2])
                    ner_tags.append(splits[3].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "lemmas": lemmas,
                    "language": language,
                    "ner_tags": ner_tags,
                }
