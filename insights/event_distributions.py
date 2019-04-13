# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

import dotdot
import numpy as np

Y_LABEL = "event density"


def main():
    signals = pd.read_csv("../data/signal_ana_20190221_105MeV.zip")

    grouped = signals.groupby("Row", sort=False).max()

    plt.hist(
        grouped["DT_layer"],
        bins=13,
        density=True,
    )
    plt.xlabel("max layer")
    plt.ylabel(Y_LABEL)
    plt.show()

    plt.hist(
        grouped["MC_hit_tu"],
        bins=18,
        density=False,
        log=True,
    )
    plt.xticks(np.arange(1.5, 19), np.arange(1, 19))
    plt.xlabel("max turn")
    plt.ylabel("number of events")
    plt.show()

    grouped = signals.groupby("Row").count()

    plt.hist(
        grouped["DT_layer"],
        bins=50,
        density=True,
    )
    plt.xlabel("number of hits")
    plt.ylabel(Y_LABEL)
    plt.show()


if __name__ == '__main__':
    main()
