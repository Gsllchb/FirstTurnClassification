# coding: utf-8
import dot

from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from util import *

LAYER = 0
MIN_SAMPLES_LEAF = 512

MIN_TPR = 0.99


def main():
    clf = joblib.load("models/DecisionTrees{}.{}.pkl".format(LAYER, MIN_SAMPLES_LEAF))
    X_test, Y_test = load_data("data/signal105MeV_test.pkl", LAYER)
    y_test = flatten(Y_test)
    y_pred = flatten(clf.predict_proba(X_test))

    logloss = log_loss(y_test, y_pred)
    print("logloss: {}".format(logloss))

    auc = roc_auc_score(y_test, y_pred)
    print("auc: {}".format(auc))

    pre_sum = tuple(itertools.accumulate(N_CELLS))
    sub_X_test = X_test[:, 2 * (pre_sum[LAYER - 1] if LAYER - 1 >= 0 else 0): 2 * pre_sum[LAYER]]
    n_null_hits = np.sum(sub_X_test == X_test.max()) // 2
    null_hit_rate = n_null_hits / np.sum(Y_test == 0)

    fprs, tprs, _ = roc_curve(y_test, y_pred)
    for fpr, tpr in zip(fprs, tprs):
        if tpr >= MIN_TPR:
            acceptance = tpr
            rejection = ((1 - fpr) - null_hit_rate) / (1 - null_hit_rate)
            msg = "first turn acceptance: {}, other turn rejection: {}."
            print(msg.format(acceptance, rejection))
            break


if __name__ == '__main__':
    main()
