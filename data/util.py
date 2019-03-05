# coding: utf-8
import array
from typing import *

import pandas as pd
import numpy as np

import dotdot
from util import *


def rename_cols(df: pd.DataFrame, rename_map: Mapping) -> None:
    df.rename(columns=rename_map, inplace=True)


def extract_events(df: pd.DataFrame, amount: int) -> pd.DataFrame:
    res = None
    for (_, event), _ in zip(df.groupby("event"), range(amount)):
        if res is None:
            res = event
        else:
            res = res.append(event, ignore_index=True)
    return res


def merge_hits(df: pd.DataFrame) -> pd.DataFrame:
    res = {
        "cell": array.array('I'),
        "layer": array.array('B'),
        "drift": array.array('f'),
        "turn": array.array('B'),
        "event": array.array('L'),
        "energy": array.array('f')
    }
    for event_id, event in df.groupby("event"):
        wires = {}
        for row in event.itertuples():
            cell = row.cell
            layer = row.layer
            drift = row.drift
            turn = row.turn
            energy = row.energy
            if (layer, cell) not in wires:
                wires[layer, cell] = (drift, energy, turn)
            else:
                prev_drift, prev_energy, prev_turn = wires[layer, cell]
                if drift < prev_drift:
                    wires[layer, cell] = (drift, prev_energy + energy, turn)
                else:
                    wires[layer, cell] = (
                        prev_drift,
                        prev_energy + energy,
                        prev_turn
                    )
        for (layer, cell), (drift, energy, turn) in wires.items():
            res["cell"].append(cell)
            res["layer"].append(layer)
            res["drift"].append(drift)
            res["turn"].append(turn)
            res["event"].append(event_id)
            res["energy"].append(energy)
    return pd.DataFrame.from_dict(res)


def group_events(df: pd.DataFrame) -> pd.DataFrame:
    n_events = df["event"].nunique()

    default_energy = df["energy"].max() + 1
    default_drift = df["drift"].max() + 1
    default_turn = 0

    initializer = {}
    for layer in range(NUM_LAYER):
        for cell in range(NUM_CELL[layer]):
            initializer[ENERGY_NAMES[layer][cell]] = np.full(
                n_events,
                default_energy,
                dtype=DTYPE_ENERGY
            )
            initializer[DRIFT_NAMES[layer][cell]] = np.full(
                n_events,
                default_drift,
                dtype=DTYPE_DRIFT
            )
            initializer[TURN_NAMES[layer][cell]] = np.full(
                n_events,
                default_turn,
                dtype=DTYPE_TURN
            )
    res = pd.DataFrame(initializer)
    del initializer

    i = None
    for i, (_, event) in enumerate(df.groupby("event")):
        for row in event.itertuples():
            res.loc[i, ENERGY_NAMES[row.layer][row.cell]] = row.energy
            res.loc[i, DRIFT_NAMES[row.layer][row.cell]] = row.drift
            res.loc[i, TURN_NAMES[row.layer][row.cell]] = row.turn
    assert i == n_events

    return res
