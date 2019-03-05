# coding: utf-8
import pandas as pd


def main():
    pd.options.display.max_columns = 100
    data = pd.read_csv("../data/signal_ana_20190221_105MeV.zip")
    print(data.describe())


if __name__ == '__main__':
    main()
