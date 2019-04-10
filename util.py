# coding: utf-8
import dot

import itertools
from typing import *

import numpy as np
from sklearn.externals import joblib

N_WIRES = 4482
N_LAYERS = 18
N_CELLS = tuple(range(198, 301, 6))
assert sum(N_CELLS) == N_WIRES

SEED = 666

DTYPE_TURN = np.int8
DTYPE_DRIFT = np.float16
DTYPE_ENERGY = np.float16

ENERGY_NAMES = tuple(tuple("energy{}_{}".format(layer, cell)
                           for cell in range(N_CELLS[layer]))
                     for layer in range(N_LAYERS))
DRIFT_NAMES = tuple(tuple("drift{}_{}".format(layer, cell)
                          for cell in range(N_CELLS[layer]))
                    for layer in range(N_LAYERS))
TURN_NAMES = tuple(tuple("turn{}_{}".format(layer, cell)
                         for cell in range(N_CELLS[layer]))
                   for layer in range(N_LAYERS))


def load_data(
        path: str,
        layer: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = joblib.load(path)
    if layer is None:
        return X, Y
    pre_sum = tuple(itertools.accumulate(N_CELLS))
    return X, Y[:, (pre_sum[layer - 1] if layer - 1 >= 0 else 0): pre_sum[layer]]


def flatten(M: np.ndarray) -> np.ndarray:
    """Flatten 2-dimension array to 1-dimension array"""
    length = M.shape[0] * M.shape[1]
    return M.reshape((length,))


def get_pa_nr_and_threshold(fprs, tprs, thresholds, min_tprs, calibration=0):
    res = []
    for min_tpr in min_tprs:
        for fpr, tpr, threshold in zip(fprs, tprs, thresholds):
            if tpr >= min_tpr:
                rejection = ((1 - fpr) - calibration) / (1 - calibration)
                res.append((tpr, rejection, threshold))
                break
    return res
