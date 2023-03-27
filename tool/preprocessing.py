import numpy as np
import pandas as pd

class Preprocessing:

    def __init__(self, data):
        """
        :param data: np.ndarray. 要处理的数据集
        """
        self.data = data

    def minmax(self, columns):
        """
        :description: min-max normalization
        :param columns: int or list. 要转换的列的序号
        """

        max = np.max(self.data[:, columns], axis=0)
        min = np.min(self.data[:, columns], axis=0)

        self.data[:, columns] = (self.data[:, columns] - min) / (max - min)

    def zscore(self, columns):
        """
        :description: z-score normalization
        :param columns: int or list. 要转换的列的序号
        """

        mean = np.mean(self.data[:, columns], axis=0)
        std = np.std(self.data[:, columns].astype(float), axis=0)

        self.data[:, columns] = (self.data[:, columns] - mean) / std

    def maxabs(self, columns):
        """
        :description: max absolute normalization
        :param columns: int or list. 要转换的列的序号
        """

        max = np.max(np.abs(self.data[:, columns]), axis=0)

        self.data[:, columns] = self.data[:, columns] / max

    def LabelEncoder(self, columns):
        """
        :description: label encoding
        :param columns: int or list. 要转换的列的序号
        :return: None
        """

        for column in columns:
            type = np.unique(self.data[:, column])
            for index, value in enumerate(type):
                self.data[:, column] = np.where(self.data[:, column] == index, value, self.data[:, column])


    def ZeroOneEncoder(self, columns):
        """
        :description: one-hot encoding
        :param columns: int or list. 要转换的列的序号
        :return: None
        """

        for column in columns:

            type = np.unique(self.data[:, column])

            if len(type) <= 2:
                self.data[:, column] = np.where(self.data[:, column] == type[0], 1, 0)
            else:
                for i in type:
                    self.data = np.insert(self.data, column + 1, np.where(self.data[:, column] == i, 1, 0), axis=1)
                self.data = np.delete(self.data, column, axis=1)

    def split(self, frac=0.7, seed=None):
        """
        :description: 划分训练集和测试集
        :param frac: float. 训练集所占比例（默认为0.7）
        :param seed: int. 随机种子（默认为None）
        :return: X_train, y_train, X_test, y_test
        """
        np.random.seed(seed)
        random_list = np.random.randint(0, self.data.shape[0], int(self.data.shape[0] * (1 - frac)))
        test = self.data[random_list]
        train = np.delete(self.data, random_list, axis=0)

        X_train = train[:, :-1]
        y_train = train[:, -1].reshape(-1, 1)
        X_test = test[:, :-1]
        y_test = test[:, -1].reshape(-1, 1)

        return X_train, y_train, X_test, y_test
