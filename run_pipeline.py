from VVTerminal import graph_extract
from Feature_extraction import feats_extract

SRC_DIRS = [
           r"C:\Users\fabia\PycharmProjects\VesselVioLib\data",
           ]

if __name__ == "__main__":

    for src in SRC_DIRS:
        graph_extract(src)
        feats_extract(src)