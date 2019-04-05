# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

from util import *


def main():
    data = pd.read_csv("../data/signal_ana_20190221_105MeV.zip")
    data = data[data["Row"] == 666]
    first_turns = data[data["MC_hit_tu"] == 1]
    other_turns = data[data["MC_hit_tu"] != 1]

    offset = 36
    for layer, n_cell in enumerate(N_CELLS):
        theta = 2 * np.pi * np.arange(n_cell) / n_cell
        r = np.full((n_cell, ), offset + layer)
        plt.polar(theta, r, "ko", markersize=0.25)

    plt.yticks(tuple())
    plt.show()


if __name__ == '__main__':
    main()
