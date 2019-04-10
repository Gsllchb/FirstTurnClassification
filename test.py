# coding: utf-8
"""Test the overall metrics of final models in hit and event levels"""
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from layerly_train import MIN_SAMPLES_LEAVES
from util import *

MIN_TPRS = (0.90, 0.95, 0.99)


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

    report_hit_level(X_test, Y_test, Y_pred, MIN_TPRS)
    report_event_level(X_test, Y_test, Y_pred, MIN_TPRS)


def report_hit_level(X_test, Y_test, Y_pred, min_tprs) -> None:
    print("Metrics report for hit level")
    print("=" * 80)
    y_test = flatten(Y_test)
    y_pred = flatten(Y_pred)
    fprs, tprs, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fprs, tprs)
    print("AUC: {}".format(auc_score))

    n_null_hits = np.sum(X_test == X_test.max()) // 2
    null_hit_rate = n_null_hits / np.sum(Y_test == 0)

    pa_nr_and_threshold1 = get_pa_nr_and_threshold(
        fprs,
        tprs,
        thresholds,
        min_tprs,
    )
    pa_nr_and_threshold2 = get_pa_nr_and_threshold(
        fprs,
        tprs,
        thresholds,
        min_tprs,
        calibration=null_hit_rate,
    )
    print("PA\tNR_raw\tNR_calibrated\tthreshold")
    for (pa, nr_raw, threshold), (_, nr_calibrated, _) in zip(pa_nr_and_threshold1, pa_nr_and_threshold2):
        print("{}\t{}\t{}\t{}".format(pa, nr_raw, nr_calibrated, threshold))

    plot_roc(fprs, tprs)


def report_event_level(X_test, Y_test, Y_pred, min_tprs) -> None:
    print("Metrics report for event level")
    print("=" * 80)
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

    fprs, tprs, thresholds = roc_curve(event_test, event_pred)
    auc_score = auc(fprs, tprs)
    print("AUC: {}".format(auc_score))
    pa_nr_and_threshold = get_pa_nr_and_threshold(
        fprs,
        tprs,
        thresholds,
        min_tprs,
    )
    print("PA\tNR\tthreshold")
    for pa, nr, threshold in pa_nr_and_threshold:
        print("{}\t{}\t{}".format(pa, nr, threshold))

    plot_roc(fprs, tprs)


def plot_roc(fprs, tprs) -> None:
    plt.plot(fprs, tprs)
    plt.plot((0, 1), (0, 1), "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


if __name__ == '__main__':
    main()
