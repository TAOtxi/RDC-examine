import numpy as np
import pandas as pd

class Preprocessing:

    def __init__(self, data):
        """
        :param data: pd.DataFrame. datasets
        """
        self.data = data

    def minmax(self, *args):
        """
        :description: min-max normalization
        :param args: pd.Series. the columns to be normalized
        """

        for column in args:
            max = np.max(column)
            min = np.min(column)

            self.data[column.name] = (column - min) / (max - min)

    def zscore(self, *args):
        """
        :description: standard normalization
        :param args: pd.Series. the columns to be normalized
        """

        for column in args:
            mean = np.mean(column)
            std = np.std(column)

            self.data[column.name] = (column - mean) / std

    def maxabs(self, *args):
        """
        :description: max absolute normalization
        :param args: pd.Series. the columns to be normalized
        """

        for column in args:
            max = np.max(np.abs(column))

            self.data[column.name] = column / max


    def LabelEncoder(self, *args):
        """
        :description: map string data to integer
        :param args: pd.Series. the columns to be encoded
        :return: None
        """

        for column in args:

            # 数据类别
            type = np.unique(column)

            replace = dict(zip(type, range(len(type))))
            self.data.replace({column.name: replace}, inplace=True)


    def ZeroOneEncoder(self, *args):
        """
        :description: one-hot encoding
        :param args: pd.Series.  the columns to be encoded
        :return: None
        """

        for column in args:

            # 数据类别
            type = np.unique(column)
            name = column.name

            if len(type) <= 2:
                self.data[name] = column.apply(lambda x: 1 if x == type[0] else 0)
            else:
                for i in type:
                    self.data[f"{name}({i})"] = column.apply(lambda x: 1 if x == i else 0)
                self.data.drop(name, axis=1, inplace=True)

    def split(self, frac=0.7, seed=None):
        """
        :description: split data into training sets and test sets
        :param frac: percentage of training sets in the whole data
        :param seed: random seed for sampling data from the whole data (default: None)
        :return: X_train, y_train, X_test, y_test
        """

        train = self.data.sample(frac=frac, random_state=seed)
        test = self.data.drop(train.index)

        X_train = train.drop('charges', axis=1).values
        y_train = train.charges.values.reshape(-1, 1)
        X_test = test.drop('charges', axis=1).values
        y_test = test.charges.values.reshape(-1, 1)

        return X_train, y_train, X_test, y_test
