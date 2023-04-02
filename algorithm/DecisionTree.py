import numpy as np
from math import log2


class DecisionTree:

    def Entropy(self, data):
        """
        :description: 求当前节点下的信息熵
        :param data: np.ndarray (n_sample, n_feature) - 数据集
        :return: float - 信息熵
        """

        # 统计标签数量
        categories, counts = np.unique(data[:, -1], return_counts=True)
        categories = dict(zip(categories, counts))

        ent = 0.0
        n_sample = data.shape[0]

        for i in categories:
            ent -= (categories[i] / n_sample) * log2(categories[i] / n_sample)

        return ent

    def InfoGain(self, data, feature):
        """
        :description: 求子集信息增益
        :param data: np.ndarray (n_sample, n_feature) - 数据集
        :param feature: int - 特征列索引
        :return: float - 信息增益
        """

        # 统计feature特征下的值
        n_sample = data.shape[0]
        vals, counts = np.unique(data[:, feature], return_counts=True)
        vals = dict(zip(vals, counts))

        # 信息增益Information Gain
        IG = self.Entropy(data)

        for val in vals:
            ent = self.Entropy(data[data[:, feature] == val])
            IG -= (vals[val] / n_sample) * ent

        return IG

    def BestFeature(self, data):
        """
        :description: 选择信息增益最大的特征
        :param data: np.ndarray (n_sample, n_feature) - 数据集
        :return: int - 信息增益最大特征的列索引
        """

        n_feature = data.shape[1] - 1
        feature = 0
        max_IG = -1

        for i in range(n_feature):

            IG = self.InfoGain(data, i)
            if IG > max_IG:
                feature = i
                max_IG = IG

        return feature

    def CreateTree(self, data, columns):
        """
        :description: 创建决策树
        :param data: np.ndarray (n_sample, n_feature) - 数据集
        :param columns: list - 列的名字
        :return: dict - 决策树
        """

        if data.shape[1] <= 1:
            return np.sort(data[:, -1])[-1]

        categories = set(data[:, -1])
        if len(categories) == len(data):
            return data[0][-1]

        feature = self.BestFeature(data)
        column = columns.pop(feature)
        tree = {column: {}}
        vals = set(data[:, feature])

        for val in vals:
            subdata = data[data[:, feature] == val]
            tree[column][val] = self.CreateTree(np.delete(subdata, obj=feature, axis=1), columns.copy())

        return tree

if __name__ == '__main__':

    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]

    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    dataSet = np.array(dataSet)
    dataSet[:, -1] = np.where(dataSet[:, -1] == 'yes', 1, 0)

    Tree = DecisionTree()
    tree = Tree.CreateTree(dataSet, labels)
    print(tree)


