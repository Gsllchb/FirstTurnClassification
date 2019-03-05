# coding: utf-8
import random

import matplotlib.pyplot as plt
import pandas as pd

import dotdot
from constants import *


def main():
    signals = pd.read_csv("../Data/signals.zip")

    n_event = max(signals.tn) + 1

    other_turns = signals[signals.turnID != 1]
    first_turns = signals[signals.turnID == 1]

    for _ in range(20):
        event_id = random.randrange(0, n_event)
        plt.scatter(
            first_turns[first_turns.tn == event_id]["mdetmt0"],
            np.log10(first_turns[first_turns.tn == event_id]["me"]),
            s=2,
            label="first turn"
        )
        plt.scatter(
            other_turns[other_turns.tn == event_id]["mdetmt0"],
            np.log10(other_turns[other_turns.tn == event_id]["me"]),
            s=2,
            label="other turn"
        )
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("log10(energy)")
        plt.title("event_id: {}".format(event_id))
        plt.show()


if __name__ == '__main__':
    main()
