# coding: utf-8
"""Test the overall metrics of final models in event level"""
from sklearn.metrics import log_loss, auc, roc_curve

from layerly_train import MIN_SAMPLES_LEAVES
from util import *

MIN_TPR = 0.90


def main():
    clfs = []
    for layer, leaf in enumerate(MIN_SAMPLES_LEAVES):
        clf = joblib.load("models/DecisionTrees{}.{}.pkl".format(layer, leaf))
        clfs.append(clf)
    X_test, Y_test = load_data("data/signal105MeV_test.pkl")

    sub_preds = []
    for clf in clfs:
        sub_preds.append(clf.predict_proba(X_test))
    Y_pred = np.hstack(sub_preds)

    assert X_test.ndim == Y_test.ndim == Y_pred.ndim
    assert X_test.shape[0] == Y_test.shape[0] == Y_pred.shape[0]
    assert X_test.shape[1] == Y_test.shape[1] * 2 == Y_pred.shape[1] * 2
    event_test = []
    event_pred = []
    placeholder = X_test.max()
    for x_test, y_test, y_pred in zip(X_test, Y_test, Y_pred):
        min_test = 1
        min_pred = 1
        for i in range(y_test.shape[0]):
            if x_test[2 * i] == placeholder:
                continue
            min_test = min(y_test[i], min_test)
            min_pred = min(y_pred[i], min_pred)
        event_test.append(min_test)
        event_pred.append(min_pred)

    logloss = log_loss(event_test, event_pred)
    print("logloss: {}".format(logloss))

    fprs, tprs, _ = roc_curve(event_test, event_pred)
    auc_score = auc(fprs, tprs)
    print("auc: {}".format(auc_score))
    for fpr, tpr in zip(fprs, tprs):
        if tpr >= MIN_TPR:
            acceptance = tpr
            rejection = 1 - fpr
            msg = "first turn event acceptance: {}, other turn event rejection: {}."
            print(msg.format(acceptance, rejection))
            break


if __name__ == '__main__':
    main()
