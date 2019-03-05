# coding: utf-8
from sklearn.model_selection import train_test_split

import dotdot
from data.util import *
from util import *


IN_PATH = "signal_ana_20190221_105MeV.zip"
TRAIN_PATH = "signal105MeV_train.pkl"
VAL_PATH = "signal105MeV_val.pkl"
TEST_PATH = "signal105MeV_test.pkl"

RENAME_MAP = {
    "DT_cell": "cell",
    "DT_layer": "layer",
    "DT_drift": "drift",
    "MC_hit_tu": "turn",
    "Row": "event",
    "MC_hit_ed": "energy",
}

TEST_SIZE = 20_000
VAL_SIZE = 20_000


def main():
    df = pd.read_csv(IN_PATH, engine="c")

    rename_cols(df, RENAME_MAP)
    df = merge_hits(df)
    transform(df)
    X, Y = group_events(df)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val,
        Y_train_val,
        test_size=VAL_SIZE,
        random_state=SEED,
    )

    joblib.dump((X_train, Y_train), TRAIN_PATH, compress=True)
    joblib.dump((X_val, Y_val), VAL_PATH, compress=True)
    joblib.dump((X_test, Y_test), TEST_PATH, compress=True)


def transform(df: pd.DataFrame) -> None:
    df["turn"] = df["turn"].map(lambda x: int(x == 1))


if __name__ == '__main__':
    main()
