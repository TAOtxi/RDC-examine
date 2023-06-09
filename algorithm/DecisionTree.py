import numpy as np
from math import log2


class DecisionTree:

    def __init__(self, data=None, FeatureNames=None, minRatio=0.01):
        """
        :param data: (n_sample, n_feature+1) - 数据集
        :param FeatureNames: list -  特征名字
        :param minRatio: float - 信息增益比的最小值
        """
        self.tree = None
        self.data = data
        self.FeatureNames = FeatureNames
        self.minRatio = minRatio
        self.best_divide = {}

    def Entropy(self, data):
        """
        :description: 求当前节点下的信息熵
        :param data: np.ndarray (n_sample, n_feature) - 数据集
        :return: float - 信息熵
        """

        # 统计标签数量
        categories, counts = np.unique(data[:, -1], return_counts=True)

        ent = 0.0
        n_sample = data.shape[0]

        for count in counts:
            ent -= (count / n_sample) * log2(count / n_sample)

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

        # 特征熵feature entropy
        FeatureEnt = self.Entropy(data[:, feature].reshape(-1, 1))

        if FeatureEnt == 0:
            return None

        return self.InfoGain(data, feature) / FeatureEnt

    def Discret(self, columns, data=None):
        """
        :description: 将连续数据离散化
        :param columns: list - 要进行离散化的列索引
        :param data: np.ndarray (n_sample, n_feature+1) - 数据集
        :return: np.ndarray - 离散化后的数据集
        """
        if data is None:
            data = self.data

        for i in columns:
            sort = data[np.argsort(data[:, i])]
            IG_max = -1

            # 比较每个划分节点下的信息增益， 选出信息增益最大的划分
            for j in range(len(data) - 1):
                divide = (sort[j, i] + sort[j + 1, i]) / 2
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
        feature = -1
        max_IGR = -1

        for i in range(n_feature):

            IGR = self.InfoGainRatio(data, i)

            if IGR is not None and IGR > max_IGR and IGR > self.minRatio:
                feature = i
                max_IGR = IGR

        return feature

    def fit(self, data=None, FeatureNames=None):
        """
        :description: 递归创建决策树
        :param data: np.ndarray (n_sample, n_feature) - 数据集
        :param FeatureNames: list - 特征名字
        :return: dict - 决策树
        """
        if data is None:
            data = self.data

        if FeatureNames is None:
            FeatureNames = self.FeatureNames.copy()

        categories, counts = np.unique(data[:, -1], return_counts=True)

        # 返回叶节点中类别最多的类别
        if data.shape[1] <= 1:
            return categories[np.argmax(counts)]

        # 正确分类，返回类别
        if len(categories) == len(data):
            return categories[0]

        feature = self.BestFeature(data)

        # 返回叶节点中类别最多的类别
        if feature == -1:
            return categories[np.argmax(counts)]

        name = FeatureNames.pop(feature)
        tree = {name: {}}
        vals = set(data[:, feature])

        for val in vals:
            subdata = data[data[:, feature] == val]

            if subdata.size == 0:
                continue
            tree[name][val] = self.fit(np.delete(subdata, obj=feature, axis=1), FeatureNames.copy())
        self.tree = tree

        return tree

    def predict(self, data, tree=None, FeatureNames=None):
        """
        :description: 通过已经生成的决策树对测试集进行分类
        :param data: np.ndarray (n_sample, n_feature+1) - 测试集
        :param tree: dict - 决策树
        :param FeatureNames: list - 特征名字
        :return: np.array - 分类结果
        """
        if tree is None:
            tree = self.tree

        if FeatureNames is None:
            FeatureNames = self.FeatureNames

        pred = []
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

            pred.append(Node)

        return np.array(pred)

    def accuracy(self, test, tree=None):
        """
        :description: 计算准确率
        :param test: np.ndarray (n_sample, n_feature+1) - 测试集
        :param tree: dict - 决策树
        :return: float - 准确率
        """
        if tree is None:
            tree = self.tree

        # 判断空数据集
        if test.size == 0:
            return 0

        pred = self.predict(test, tree)

        return np.sum(pred == test[:, -1]) / len(pred)

    # 后剪枝
    def PostPruning(self, train, test, tree=None, FeatureNames=None):
        """
        :description: 递归进行后剪枝
        :param train: np.ndarray (n_sample, n_feature+1) - 训练集
        :param test: np.ndarray (n_sample, n_feature+1) - 测试集
        :param tree: dict - 决策树
        :param FeatureNames: list - 特征名字
        :return: dict - 剪枝后的决策树
        """

        if tree is None:
            tree = self.tree

        if FeatureNames is None:
            FeatureNames = self.FeatureNames

        feature = list(tree.keys())[0]
        column = FeatureNames.index(feature)

        for val in tree[feature].keys():
            subtree = tree[feature][val]

            if isinstance(subtree, dict):
                subtrain = train[train[:, column] == val]
                subtest = test[test[:, column] == val]

                # 递归调用self.PostPruning对子树进行后剪枝
                Node = self.PostPruning(subtrain, subtest, subtree, FeatureNames)

            # 到达叶子节点时，进行剪枝操作
            if test.size != 0 and not isinstance(subtree, dict):

                # 备份子树节点
                backup = tree[feature][val]

                # 剪枝前
                before = self.accuracy(test, tree)

                # 进行剪枝操作，将当前节点替换为叶子节点
                categories, counts = np.unique(train[:, -1], return_counts=True)
                label = categories[np.argmax(counts)]
                tree[feature][val] = label

                # 剪枝后
                after = self.accuracy(test, tree)

                # 如果剪枝后的准确率变差，则还原子树节点
                if before > after:
                    tree[feature][val] = backup

        return tree
