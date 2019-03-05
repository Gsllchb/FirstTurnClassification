# coding: utf-8
import itertools
from typing import *

import numpy as np
from sklearn.externals import joblib

NUM_WIRE = 4482
NUM_LAYER = 18
NUM_CELL = tuple(range(198, 301, 6))
assert sum(NUM_CELL) == NUM_WIRE

SEED = 666

DTYPE_TURN = np.int8
DTYPE_DRIFT = np.float32
DTYPE_ENERGY = np.float32

ENERGY_NAMES = tuple(tuple("energy{}_{}".format(layer, cell)
                           for cell in range(NUM_CELL[layer]))
                     for layer in range(NUM_LAYER))
DRIFT_NAMES = tuple(tuple("drift{}_{}".format(layer, cell)
                          for cell in range(NUM_CELL[layer]))
                    for layer in range(NUM_LAYER))
TURN_NAMES = tuple(tuple("turn{}_{}".format(layer, cell)
                         for cell in range(NUM_CELL[layer]))
                   for layer in range(NUM_LAYER))


def load_data(
        path: str,
        layer: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    data = joblib.load(path)
    features = itertools.chain(*ENERGY_NAMES, *DRIFT_NAMES)
    if layer is None:
        targets = itertools.chain(*TURN_NAMES)
    else:
        targets = TURN_NAMES[layer]
    X = data.loc[:, features].values
    Y = data.loc[:, targets].values
    return X, Y


def flatten(M: np.ndarray) -> np.ndarray:
    """Flatten 2-dimension array to 1-dimension array"""
    length = M.shape[0] * M.shape[1]
    return M.reshape((length,))
