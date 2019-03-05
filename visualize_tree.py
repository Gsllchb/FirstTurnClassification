# coding: utf-8
import graphviz
from sklearn.externals import joblib
from sklearn.tree import export_graphviz

from util import *


DECISION_TREE = joblib.load("Models/DecisionTrees0.0.001.pkl").estimators_[66]


def main():
    output_pdf(DECISION_TREE, ".temp")


def output_pdf(decision_tree, file: str) -> None:
    features = itertools.chain(*ENERGY_NAMES, *DRIFT_NAMES)
    dot_data = export_graphviz(
        decision_tree,
        out_file=None,
        proportion=True,
        rounded=True,
        feature_names=features,
    )
    graphviz.Source(dot_data).render(file, format="png")


if __name__ == '__main__':
    main()
