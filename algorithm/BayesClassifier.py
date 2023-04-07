import numpy as np


class Bayes:

    def __init__(self):

        self.prob = {}
        self.mean = {}
        self.std = {}
        self.DiscColumns = None
        self.ContiColumns = None

    def p(self, category, column, val):
        """
        :description: 求连续型数据在概率密度函数上的取值
        :param category: float - 数据值
        :param column: int - 列索引
        :param val: 列的值
        :return: 略
        """
        mean = self.mean[category][column]
        std = self.std[category][column]

        return 1 / (np.sqrt(2 * np.pi)) * np.exp(-(np.square(val - mean)) / (2 * np.square(std)))

    def fit(self, data, DiscColumns=None, ContiColumns=None):
        """
        :description: 打表
        :param data: np.ndarray (n_sample, n_feature+1) - 数据集
        :param DiscColumns: list - 离散数据的列索引
        :param ContiColumns: -ist - 连续数据的列索引
        :return: None
        """
        if DiscColumns is None:
            DiscColumns = np.arange(data.shape[1] - 1)

        self.DiscColumns = DiscColumns
        self.ContiColumns = ContiColumns
        self.prob = {}
        n_sample = data.shape[0]
        categories, counts = np.unique(data[:, -1], return_counts=True)

        for category, count in zip(categories, counts):
            self.prob[category] = {}
            self.prob[category]['prob'] = count / n_sample
            subdata = data[data[:, -1] == category]
            sub_sample = subdata.shape[0]

            for column in DiscColumns:
                vals, counts = np.unique(subdata[:, column], return_counts=True)
                self.prob[category][column] = dict(zip(vals, counts / sub_sample))

            self.mean[category] = {}
            self.std[category] = {}
            for column in ContiColumns:
                self.mean[category][column] = subdata[:, column].mean()
                self.std[category][column] = subdata[:, column].std()

    def pred(self, data):
        """
        :description: 预测
        :param data: np.ndarray (n_sample, n_feature+1) - 测试集
        :return: np.ndarray (n_sample, 1) - 预测值
        """
        pred = []

        for i in data:
            MaxProb = 0.0
            exp = None
            for category in self.prob.keys():
                prob = self.prob[category]['prob']

                for col in self.DiscColumns:
                    prob *= self.prob[category][col][i[col]]

                for col in self.ContiColumns:
                    prob *= self.p(category, col, i[col])

                if prob > MaxProb:
                    MaxProb = prob
                    exp = category

            pred.append(exp)

        return np.array(pred)

    def accuracy(self, X, y_true):
        """
        :description: 计算准确率
        :param X: np.ndarray (n_sample, n_feature) - 测试集
        :param y_true: np.ndarray (n_sample,) - 真实值
        :return: float - 准确率
        """
        y_pred = self.pred(X)

        return (y_pred == y_true).sum() / y_true.size

