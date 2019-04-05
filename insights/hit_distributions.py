# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

import dotdot
from util import *

BIN_NUM = 50


def main():
    data = pd.read_csv("../data/signal_ana_20190221_105MeV.zip")
    first_turns = data[data["MC_hit_tu"] == 1]
    other_turns = data[data["MC_hit_tu"] != 1]
    first_turns_label = "first_turns"
    other_turns_label = "other_turns"

    y_label = "hit density"
    density = True

    plt.hist(
        np.log2(first_turns["MC_hit_ed"]),
        BIN_NUM,
        label=first_turns_label,
        density=density,
        alpha=0.5
    )
    plt.hist(
        np.log2(other_turns["MC_hit_ed"]),
        BIN_NUM,
        label=other_turns_label,
        density=density,
        alpha=0.5
    )
    plt.legend()
    plt.xlabel("deposit energy (log2 scale)")
    plt.ylabel(y_label)
    plt.show()

    plt.hist(
        first_turns["DT_drift"],
        BIN_NUM,
        label=first_turns_label,
        density=density,
        alpha=0.5
    )
    plt.hist(
        other_turns["DT_drift"],
        BIN_NUM,
        label=other_turns_label,
        density=density,
        alpha=0.5
    )
    plt.legend()
    plt.xlabel("drift time")
    plt.ylabel(y_label)
    plt.show()

    plt.hist(
        first_turns["DT_layer"],
        N_LAYERS,
        label=first_turns_label,
        density=density,
        alpha=0.5
    )
    plt.hist(
        other_turns["DT_layer"],
        N_LAYERS,
        label=other_turns_label,
        density=density,
        alpha=0.5
    )
    plt.legend()
    plt.xlabel("layer")
    plt.ylabel(y_label)
    plt.show()


if __name__ == '__main__':
    main()
