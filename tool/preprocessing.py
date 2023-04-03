import numpy as np

class Preprocessing:

    def minmax(self, data, columns):
        """
        :description: 极差标准化
        :param data: np.ndarray - 数据集
        :param columns: int or list - 要转换的列的序号
        :return: np.ndarray - 转换后的数据集
        """

        max = np.max(data[:, columns], axis=0)
        min = np.min(data[:, columns], axis=0)

        data[:, columns] = (data[:, columns] - min) / (max - min)
        return data

    def zscore(self, data, columns):
        """
        :description: 标准差标准化
        :param data: np.ndarray - 数据集
        :param columns: int or list - 要转换的列的序号
        :return: np.ndarray - 转换后的数据集
        """

        mean = np.mean(data[:, columns], axis=0)
        std = np.std(data[:, columns].astype(float), axis=0)

        data[:, columns] = (data[:, columns] - mean) / std
        return data

    def maxabs(self, data, columns):
        """
        :description: 极大值绝对值标准化
        :param data: np.ndarray - 数据集
        :param columns: int or list - 要转换的列的序号
        :return: np.ndarray - 转换后的数据集
        """

        maxabs = np.max(np.abs(data[:, columns]), axis=0)

        data[:, columns] = data[:, columns] / maxabs
        return data

    def LabelEncoder(self, data, columns):
        """
        :description: 序号编码
        :param data: np.ndarray - 数据集
        :param columns: list - 要转换的列的序号
        :return: np.ndarray - 转换后的数据集
        """

        for column in columns:
            type = np.unique(data[:, column])
            for index, value in enumerate(type):
                data[:, column][data[:, column] == value] = index

        return data

    def OneHotEncoder(self, data, columns):
        """
        :description: 独热编码
        :param data: np.ndarray - 数据集
        :param columns: list - 要转换的列的序号
        :return: np.ndarray - 转换后的数据集
        """

        for column in columns:
            type = np.unique(data[:, column])

            if len(type) <= 2:
                data[:, column] = np.where(data[:, column] == type[0], 1, 0)
            else:
                for i in type:
                    data = np.insert(data, column + 1, np.where(data[:, column] == i, 1, 0), axis=1)
                data = np.delete(data, column, axis=1)

        return data

    def split(self, data, frac=0.7, seed=None, split_XY=True, convert=True):
        """
        :description: 划分训练集和测试集
        :param data: np.ndarray - 数据集
        :param frac: float - 训练集所占比例（默认为0.7）
        :param seed: float - 随机种子（默认为None）
        :param split_XY: bool - 默认把数据和标签分开，只返回train和test
        :param convert: bool - 默认把数据转换成np.float64型
        :return: np.ndarray * 4 - 划分好的训练集X_train, 训练集标签y_train, 测试集X_test, 测试集标签y_test
        """

        if convert:
            data = data.astype(np.float64)

        n_samples = data.shape[0]
        np.random.seed(seed)
        random_index = np.random.permutation(range(n_samples))
        train = data[random_index[:int(n_samples * frac)]]
        test = data[random_index[int(n_samples * frac):]]

        if split_XY:
            X_train = train[:, :-1]
            y_train = train[:, -1].reshape(-1, 1)
            X_test = test[:, :-1]
            y_test = test[:, -1].reshape(-1, 1)

            return X_train, y_train, X_test, y_test

        else:
            return train, test
