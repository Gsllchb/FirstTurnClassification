# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

import dotdot

Y_LABEL = "event density"


def main():
    signals = pd.read_csv("../Data/signals.zip")

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
        density=True,
    )
    plt.xlabel("max turn id")
    plt.ylabel(Y_LABEL)
    plt.show()

    grouped = signals.groupby("Row").count()

    plt.hist(
        grouped["DT_layer"],
        bins=50,
        density=True,
    )
    plt.xlabel("hit amount")
    plt.ylabel(Y_LABEL)
    plt.show()


if __name__ == '__main__':
    main()
