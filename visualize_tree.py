# coding: utf-8
import graphviz
from sklearn.tree import export_graphviz

from util import *

DECISION_TREE = joblib.load("Models/DecisionTrees0.248.pkl").estimators_[30]


def main():
    output_pdf(DECISION_TREE, ".temp")


def output_pdf(decision_tree, file: str) -> None:
    dot_data = export_graphviz(
        decision_tree,
        out_file=None,
        proportion=True,
        rounded=True,
    )
    graphviz.Source(dot_data).render(file, format="png")


if __name__ == '__main__':
    main()
