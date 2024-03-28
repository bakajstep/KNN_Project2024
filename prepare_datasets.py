from ourdatasets.wikiann.wikiann_cs import prepare_wikiann
from ourdatasets.cnec2_extended.cnec2_extended import get_cnec2_extended


def main():
    prepare_wikiann()
    get_cnec2_extended()

if __name__ == '__main__':
    main()