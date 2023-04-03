import numpy as np
from math import log2


class DecisionTree:

    def __init__(self, data=None, FeatureNames=None):
        """
        :param data: (n_sample, n_feature+1) - 数据集
        :param FeatureNames: list -  特征名字
        """
        self.tree = None
        self.data = data
        self.FeatureNames = FeatureNames
        self.best_divide = {}

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

        for i in categories.keys():
            ent -= (categories[i] / n_sample) * log2(categories[i] / n_sample)

        return ent

    def InfoGain(self, data, feature):
        """
        :description: 求子集信息增益
        :param data: np.ndarray (n_sample, n_feature+1) - 数据集
        :param feature: int - 特征列索引
        :return: float - 信息增益
        """

        # 统计feature特征下的值
        n_sample = data.shape[0]
        vals, counts = np.unique(data[:, feature], return_counts=True)
        vals = dict(zip(vals, counts))

        # 信息增益Information Gain
        IG = self.Entropy(data)

        for val in vals.keys():
            ent = self.Entropy(data[data[:, feature] == val])
            IG -= (vals[val] / n_sample) * ent

        return IG

    def InfoGainRatio(self, data, feature):
        """
        :description: 求子集信息增益比
        :param data: np.ndarray (n_sample, n_feature+1) - 数据集
        :param feature: int - 特征列索引
        :return: float - 信息增益比
        """
        n_sample = data.shape[0]
        vals, counts = np.unique(data[:, feature], return_counts=True)
        vals = dict(zip(vals, counts))

        # 特征熵feature entropy
        FeatureEnt = 0.0
        for val in vals.keys():
            FeatureEnt -= (vals[val] / n_sample) * log2(vals[val] / n_sample)

        return self.InfoGain(data, feature) / FeatureEnt

    def Discret(self, columns, data=None):
        """
        :description: 将连续数据离散化
        :param columns: list - 要进行离散化的列索引
        :param data: np.ndarray (n_sample, n_feature+1) - 数据集
        :return: np.ndarray - 离散化后的数据集
        """
        if data is None:
            if self.data is None:
                raise '空数据集...'
            data = self.data

        for i in columns:
            sort = data[np.argsort(data[:, i])]
            IG_max = -1

            # 比较每个划分节点下的信息增益， 选出信息增益最大的划分
            for j in range(len(data) - 1):
                divide = (sort[j, i] + sort[j + 1, i] + 1) / 2
                sort[:, i] = np.where(sort[:, i] > divide, 1, 0)
                IG = self.InfoGain(data=sort, feature=i)

                if IG > IG_max:
                    IG_max = IG
                    self.best_divide[i] = divide
            data[:, i] = np.where(data[:, i] > self.best_divide[i], 1, 0)
        self.data = data
        return data

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

            IG = self.InfoGainRatio(data, i)
            if IG > max_IG:
                feature = i
                max_IG = IG

        return feature

    def CreateTree(self, data=None, FeatureNames=None):
        """
        :description: 递归创建决策树
        :param data: np.ndarray (n_sample, n_feature) - 数据集
        :param FeatureNames: list - 特征名字
        :return: dict - 决策树
        """
        if data is None:
            if self.data is None:
                raise '空数据集...'
            data = self.data

        if FeatureNames is None:
            if self.FeatureNames is None:
                raise '空名字...'
            FeatureNames = self.FeatureNames.copy()

        categories, counts = np.unique(data[:, -1], return_counts=True)

        # 返回叶节点中类别最多的类别
        if data.shape[1] <= 1:
            return categories[np.argmax(counts)]

        # 正确分类，返回类别
        if len(categories) == len(data):
            return categories[0]

        feature = self.BestFeature(data)
        name = FeatureNames.pop(feature)
        tree = {name: {}}
        vals = set(data[:, feature])

        for val in vals:
            subdata = data[data[:, feature] == val]
            tree[name][val] = self.CreateTree(np.delete(subdata, obj=feature, axis=1), FeatureNames.copy())
        self.tree = tree

        return tree

    def classify(self, data, tree=None, FeatureNames=None):
        """
        :description: 通过已经生成的决策树对测试集进行分类
        :param data: np.ndarray (n_sample, n_feature+1) - 测试集
        :param tree: dict - 决策树
        :param FeatureNames: list - 特征名字
        :return: list - 分类结果
        """
        if tree is None:
            tree = self.tree

        if FeatureNames is None:
            FeatureNames = self.FeatureNames

        out = []
        for sample in data:
            feature = list(tree.keys())[0]
            value = sample[FeatureNames.index(feature)]
            Node = tree[feature].get(value)

            while isinstance(Node, dict):
                feature = list(Node.keys())[0]
                value = sample[FeatureNames.index(feature)]
                if Node[feature].get(value) is None:
                    Node = list(Node[feature].values())[0]
                else:
                    Node = Node[feature].get(value)

            out.append(Node)

        return out

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

    Tree = DecisionTree(dataSet, labels)
    tree = Tree.CreateTree()
    print(tree)
    classify = np.array(Tree.classify(dataSet))
    print(classify == dataSet[:, -1])

