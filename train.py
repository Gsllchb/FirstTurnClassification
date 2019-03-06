# coding: utf-8
import datetime

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

from util import *

LAYER = 0

# MIN_SAMPLES_LEAVES = (
#     266,
#     306,
#     197,
#     205,
# )
MIN_SAMPLES_LEAVES = (
    512,
)
MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAVES[LAYER]

LOG_FILE = "train.log"
DUMP_PATH = "models/DecisionTrees{}.{}.pkl".format(LAYER, MIN_SAMPLES_LEAF)
LOG_FMT = """
{time}
layer: {layer}
min_samples_leaf: {min_samples_leaf}
{clf}
logloss: {logloss}
auc: {auc}
"""

N_JOBS = -1


def main():
    print("loading data...")
    X_train, Y_train = load_data("data/signal105MeV_train.pkl", LAYER)

    clf = OneVsRestClassifier(
        DecisionTreeClassifier(
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight="balanced",
            random_state=SEED,
        ),
        n_jobs=N_JOBS,
    )

    print("training model...")
    clf.fit(X_train, Y_train)

    X_val, Y_val = load_data("data/signal105MeV_val.pkl", LAYER)
    y_val = flatten(Y_val)
    y_pred = flatten(clf.predict_proba(X_val))
    logloss = log_loss(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    print("logloss: {}".format(logloss))
    print("auc: {}".format(auc))

    print("dumping...")
    joblib.dump(clf, DUMP_PATH, compress=True)

    print("logging...")
    content = LOG_FMT.format(
        time=datetime.datetime.now(),
        layer=LAYER,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        clf=clf,
        logloss=logloss,
        auc=auc,
    )
    with open(LOG_FILE, mode="a", encoding="utf-8") as f:
        f.write(content)

    print("All done!!!\a")


if __name__ == '__main__':
    main()
