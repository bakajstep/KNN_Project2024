from parsers.cnec2_extended.cnec2_extended import get_cnec2_extended
from parsers.medival.medival_parser import prepare_medival
from parsers.poner.poner import prepare_poner
from parsers.slavic.slavic_bsnlp import prepare_slavic
from parsers.wikiann_cs.wikiann_cs import prepare_wikiann


def prepare_datasets_to_conll(output_dir):
    prepare_wikiann(output_dir, "wikiann")
    prepare_poner(output_dir, "poner")
    get_cnec2_extended(
        "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3493/"
        "cnec2.0_extended.zip",
        output_dir,
        "cnec_dataset")
    prepare_slavic(
        "https://bsnlp.cs.helsinki.fi/bsnlp-2021/data/bsnlp2021_train_r1.zip",
        "https://bsnlp.cs.helsinki.fi/bsnlp-2021/data/bsnlp2021_test_v5.zip",
        output_dir,
        "slavic_dataset")
    prepare_medival(
        'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5024/'
        'named-entity-recognition-annotations-small.zip?sequence=2&isAllowed=y',
        output_dir, "medival")


if __name__ == '__main__':
    prepare_datasets_to_conll("./custom")
