from parsers.cnec2_extended.cnec2_extended import get_cnec2_extended
from parsers.medival.medival_parser import prepare_medival
from parsers.poner.poner import prepare_poner
from parsers.slavic.slavic_bsnlp import prepare_slavic
from parsers.wikiann_cs.wikiann_cs import prepare_wikiann


def main(output_dir):
    prepare_wikiann(output_dir, "wikiann")
    prepare_poner(output_dir, "poner")
    get_cnec2_extended(
        "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3493/"
        "cnec2.0_extended.zip",
        output_dir,
        "cnec_dataset")
    prepare_slavic(
        "https://bsnlp.cs.helsinki.fi/bsnlp-2019/TRAININGDATA_BSNLP_2019_shared_task.zip",
        "https://bsnlp.cs.helsinki.fi/bsnlp-2019/TESTDATA_BSNLP_2019_shared_task.zip",
        output_dir,
        "slavic_dataset")
    prepare_medival(
        'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5024/'
        'named-entity-recognition-annotations-small.zip?sequence=2&isAllowed=y',
        output_dir, "medival_dataset")


if __name__ == '__main__':
    main("./custom")
