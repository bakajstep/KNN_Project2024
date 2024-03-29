from ourdatasets.slavic.slavic_bsnlp import prepare_slavic
from ourdatasets.wikiann.wikiann_cs import prepare_wikiann
from ourdatasets.cnec2_extended.cnec2_extended import get_cnec2_extended


def main():
    prepare_wikiann()
    get_cnec2_extended()
    prepare_slavic("https://bsnlp.cs.helsinki.fi/bsnlp-2019/TRAININGDATA_BSNLP_2019_shared_task.zip", "train.conll",
                     "cs")
    prepare_slavic("https://bsnlp.cs.helsinki.fi/bsnlp-2019/TESTDATA_BSNLP_2019_shared_task.zip", "test.conll", "cs")

if __name__ == '__main__':
    main()