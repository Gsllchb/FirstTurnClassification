# coding: utf-8
"""随机选择10个事件并将其横截面画出"""
import random

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import pandas as pd

import dotdot
from constants import *

MAX_X = 80
MAX_Y = 80
MIN_X = -80
MIN_Y = -80


def main():
    # load data
    signals = pd.read_csv("../Data/signals.zip")

    max_event_id = max(signals.tn)

    # rescale
    signals.xe0 = (signals.xe0 - MIN_X) / (MAX_X - MIN_X)
    signals.xe1 = (signals.xe1 - MIN_X) / (MAX_X - MIN_X)
    signals.ye0 = (signals.ye0 - MIN_Y) / (MAX_Y - MIN_Y)
    signals.ye1 = (signals.ye1 - MIN_Y) / (MAX_Y - MIN_Y)

    other_turns = signals[signals.turnID != 1]
    first_turns = signals[signals.turnID == 1]

    # draw 10 events
    for _ in range(20):
        fig = plt.figure(figsize=(6, 6))
        event_id = random.randint(0, max_event_id)
        sub_first_turns = first_turns[first_turns.tn == event_id]
        sub_other_turns = other_turns[other_turns.tn == event_id]
        ls = []

        # draw other_turns first
        for xe0, ye0, xe1, ye1 in zip(
                sub_other_turns.xe0,
                sub_other_turns.ye0,
                sub_other_turns.xe1,
                sub_other_turns.ye1
        ):
            l = lines.Line2D(
                (xe0, xe1),
                (ye0, ye1),
                transform=fig.transFigure,
                figure=fig,
                linewidth=1,
                color="b"
            )
            ls.append(l)

        for xe0, ye0, xe1, ye1 in zip(
                sub_first_turns.xe0,
                sub_first_turns.ye0,
                sub_first_turns.xe1,
                sub_first_turns.ye1
        ):
            l = lines.Line2D(
                (xe0, xe1),
                (ye0, ye1),
                transform=fig.transFigure,
                figure=fig,
                linewidth=1,
                color="r"
            )
            ls.append(l)

        fig.lines.extend(ls)
        fig.text(0.01, 0.01, "event_id: {}".format(event_id))
        plt.show()


if __name__ == '__main__':
    main()
