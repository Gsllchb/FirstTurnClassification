# coding: utf-8
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
    df = group_events(df)

    df.reindex(np.random.permutation(df.index), copy=False)
    test_set = df.iloc[:TEST_SIZE]
    val_set = df.iloc[TEST_SIZE: TEST_SIZE + VAL_SIZE]
    train_set = df.iloc[TEST_SIZE + VAL_SIZE:]

    joblib.dump(train_set, TRAIN_PATH, compress=True)
    joblib.dump(val_set, VAL_PATH, compress=True)
    joblib.dump(test_set, TEST_PATH, compress=True)


def transform(df: pd.DataFrame) -> None:
    df["turn"] = df["turn"].map(lambda x: int(x == 1))


if __name__ == '__main__':
    main()
