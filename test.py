# coding: utf-8
from sklearn.externals import joblib
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

import dot
from util import *

LAYER = 0
MIN_SAMPLES_LEAF = 512

MIN_TPR = 0.95


def main():
    clf = joblib.load("models/DecisionTrees{}.{}.pkl".format(LAYER, MIN_SAMPLES_LEAF))
    X_test, Y_test = load_data("data/signal105MeV_test.pkl", LAYER)
    y_test = flatten(Y_test)
    y_pred = flatten(clf.predict_proba(X_test))

    logloss = log_loss(y_test, y_pred)
    print("logloss: {}".format(logloss))

    auc = roc_auc_score(y_test, y_pred)
    print("auc: {}".format(auc))

    fprs, tprs, _ = roc_curve(y_test, y_pred)
    for fpr, tpr in zip(fprs, tprs):
        if tpr >= MIN_TPR:
            msg = "first turn acceptance: {}, others rejection: {}."
            print(msg.format(tpr, 1 - fpr))
            break


if __name__ == '__main__':
    main()
