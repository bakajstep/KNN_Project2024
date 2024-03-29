from ourdatasets.cnec2_extended.cnec2_extended import get_cnec2_extended
from ourdatasets.slavic.slavic_bsnlp import prepare_slavic
from ourdatasets.wikiann.wikiann_cs import prepare_wikiann


def main():
    prepare_wikiann()
    get_cnec2_extended()
    prepare_slavic(
        "https://bsnlp.cs.helsinki.fi/bsnlp-2019/TRAININGDATA_BSNLP_2019_shared_task.zip",
        "train.conll",
        "cs",
        ["training_pl_cs_ru_bg_rc1/annotated/cs/"],
        ["training_pl_cs_ru_bg_rc1/raw/cs/"]
    )

    prepare_slavic(
        "https://bsnlp.cs.helsinki.fi/bsnlp-2019/TESTDATA_BSNLP_2019_shared_task.zip",
        "test.conll",
        "cs",
        ["annotated/nord_stream/cs/", "annotated/ryanair/cs/"],
        ["raw/nord_stream/cs/", "raw/ryanair/cs/"]
    )


if __name__ == '__main__':
    main()
