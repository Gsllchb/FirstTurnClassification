# coding: utf-8
import array

import pandas as pd

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


def group_events(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    n_events = df["event"].nunique()

    default_energy = df["energy"].max() + 1
    default_drift = df["drift"].max() + 1
    default_feature = max(default_energy, default_drift)
    default_turn = 0

    assert DTYPE_DRIFT == DTYPE_ENERGY
    features = np.full(
        (n_events, 2 * N_WIRES),
        fill_value=default_feature,
        dtype=DTYPE_ENERGY
    )
    targets = np.full(
        (n_events, N_WIRES),
        fill_value=default_turn,
        dtype=DTYPE_TURN
    )

    pre_sum = tuple(itertools.accumulate(N_CELLS))
    i = None
    for i, (_, event) in enumerate(df.groupby("event")):
        for row in event.itertuples():
            cell_index = (pre_sum[row.layer - 1] if row.layer - 1 >= 0 else 0) + row.cell
            features[i, cell_index * 2] = row.energy
            features[i, cell_index * 2 + 1] = row.drift
            targets[i, cell_index] = row.turn
    assert i == n_events - 1

    return features, targets
