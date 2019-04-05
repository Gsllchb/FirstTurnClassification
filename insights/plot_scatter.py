# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

from util import *


def main():
    data = pd.read_csv("../data/signal_ana_20190221_105MeV.zip")
    data = data[data["Row"] < 32]
    first_turns = data[data["MC_hit_tu"] == 1]
    other_turns = data[data["MC_hit_tu"] != 1]
    first_turns_label = "first turn"
    other_turns_label = "other turn"

    plt.scatter(
        first_turns["DT_drift"],
        np.log2(first_turns["MC_hit_ed"]),
        label=first_turns_label,
        s=2,
    )
    plt.scatter(
        other_turns["DT_drift"],
        np.log2(other_turns["MC_hit_ed"]),
        label=other_turns_label,
        s=2,
    )
    plt.legend()
    plt.xlabel("drift time")
    plt.ylabel("deposit energy (log2 scale)")
    plt.show()


if __name__ == '__main__':
    main()
