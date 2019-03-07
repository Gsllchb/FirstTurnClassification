# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

import dotdot
from util import *

BIN_NUM = 50


def main():
    data = pd.read_csv("../data/signal_ana_20190221_105MeV.zip")
    positive = data[data["MC_hit_tu"] == 1]
    negative = data[data["MC_hit_tu"] != 1]
    positive_label = "positive"
    negative_label = "negative"

    for density in (True, False):
        y_label = "hit density" if density else "hit"

        plt.hist(
            np.log2(positive["MC_hit_ed"]),
            BIN_NUM,
            label=positive_label,
            density=density,
            alpha=0.5
        )
        plt.hist(
            np.log2(negative["MC_hit_ed"]),
            BIN_NUM,
            label=negative_label,
            density=density,
            alpha=0.5
        )
        plt.legend()
        plt.xlabel("log2(energy)")
        plt.ylabel(y_label)
        plt.show()

        plt.hist(
            positive["DT_drift"],
            BIN_NUM,
            label=positive_label,
            density=density,
            alpha=0.5
        )
        plt.hist(
            negative["DT_drift"],
            BIN_NUM,
            label=negative_label,
            density=density,
            alpha=0.5
        )
        plt.legend()
        plt.xlabel("drift")
        plt.ylabel(y_label)
        plt.show()

        plt.hist(
            positive["DT_layer"],
            N_LAYERS,
            label=positive_label,
            density=density,
            alpha=0.5
        )
        plt.hist(
            negative["DT_layer"],
            N_LAYERS,
            label=negative_label,
            density=density,
            alpha=0.5
        )
        plt.legend()
        plt.xlabel("layer")
        plt.ylabel(y_label)
        plt.show()


if __name__ == '__main__':
    main()
