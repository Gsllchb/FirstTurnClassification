# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

import dotdot
from util import *

BIN_NUM = 50


def main():
    data = pd.read_csv("../data/signal_ana_20190221_105MeV.zip")

    first_turns_label = "first_turns"
    other_turns_label = "other_turns"

    y_label = "hit density"
    density = True

    for _, event in data.groupby("Row"):
        first_turns = event[event["MC_hit_tu"] == 1]
        other_turns = event[event["MC_hit_tu"] != 1]
        if other_turns.empty:
            continue
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
        plt.xlabel("log2(energy)")
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
        plt.xlabel("drift")
        plt.ylabel(y_label)
        plt.show()


if __name__ == '__main__':
    main()
