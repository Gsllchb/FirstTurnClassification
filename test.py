# coding: utf-8
import dot

from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from util import *
from layer_train import MIN_SAMPLES_LEAVES

MIN_TPR = 0.90


def main():
    clfs = []
    for layer, leaf in enumerate(MIN_SAMPLES_LEAVES):
        clf = joblib.load("models/DecisionTrees{}.{}.pkl".format(layer, leaf))
        clfs.append(clf)
    X_test, Y_test = load_data("data/signal105MeV_test.pkl")
    y_test = flatten(Y_test)

    sub_preds = []
    for layer, clf in enumerate(clfs):
        sub_preds.append(clf.predict_proba(X_test))
    Y_pred = np.hstack(sub_preds)
    y_pred = flatten(Y_pred)

    logloss = log_loss(y_test, y_pred)
    print("logloss: {}".format(logloss))

    auc = roc_auc_score(y_test, y_pred)
    print("auc: {}".format(auc))

    n_null_hits = np.sum(X_test == X_test.max()) // 2
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
