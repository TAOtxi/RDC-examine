import numpy as np


class Evaluation:

    def __init__(self, X, theta, y_true, threshold=None):
        """
        :description: 评估模型. (MAE, MSE, RMSE, R2, precision, recall, accuracy, F1, ROC)
        :param X: np.ndarray (n_sample, n_feature) - 特征矩阵.
        :param theta: np.ndarray (n_feature, 1) - 模型参数.
        :param y_true: np.ndarray (n_sample, 1) - 真实标签.
        :param threshold: float - 分类模型的阈值
        """

        self.y_true = y_true
        self.n_samples = y_true.shape[0]
        self.threshold = threshold
        y = np.insert(X, 0, 1, axis=1) @ theta
        self.sigmoid = 1 / (1 + np.exp(-y))

        if threshold is not None:
            self.y_pred = np.where(self.sigmoid >= threshold, 1, 0)
        else:
            self.y_pred = y

    def MAE(self):
        """
        :description: 求平均绝对误差
        :return: float - 平均绝对误差
        """
        return np.mean(np.abs(self.y_pred - self.y_true)) / self.n_samples

    def MSE(self):
        """
        :description: 求均方误差
        :return: float - 均方误差
        """
        diff = self.y_pred - self.y_true

        # diff.T @ diff.shape == (1, 1)
        return (diff.T @ diff / self.n_samples)[0][0]

    def RMSE(self):
        """
        :description: 求均方根误差
        :return: float - 均方根误差
        """
        return np.sqrt(self.MSE())

    def R2(self):
        """
        :description: 求模型的决定系数
        :return: float - 决定系数
        """
        y_mean = np.mean(self.y_true)
        diff_true = self.y_pred - self.y_true
        diff_mean = self.y_pred - y_mean
        return 1 - ((diff_true.T @ diff_true) / (diff_mean.T @ diff_mean))[0][0]

    def precision(self, threshold=None):
        """
        :description: 分类求模型的精确率 TP / (TP + FP)
        :param threshold: float - 分类模型的阈值
        :return: float - 精确率
        """

        if threshold is None:
            if self.threshold is None:
                threshold = 0.5
            else:
                threshold = self.threshold

        self.y_pred = np.where(self.sigmoid >= threshold, 1, 0)

        diff = self.y_pred - self.y_true
        TP = np.sum(diff == 0)
        FP = np.sum(diff == 1)

        return TP / (TP + FP)

    def recall(self, threshold=None):
        """
        :description: 分类模型的召回率 TP / (TP + FN)
        :param threshold: float - 分类模型的阈值
        :return: float - 召回率
        """

        if threshold is None:
            if self.threshold is None:
                threshold = 0.5
            else:
                threshold = self.threshold

        self.y_pred = np.where(self.sigmoid >= threshold, 1, 0)

        same_index = self.y_pred == self.y_true
        diff = self.y_pred - self.y_true

        TP = np.sum(self.y_true[same_index] == 1)
        FN = np.sum(diff == -1)

        return TP / (TP + FN)

    def accuracy(self, threshold=None):
        """
        :description: 求分类模型的准确率 (TP + TN) / (TP + TN + FP + FN)
        :param threshold: float - 分类模型的阈值
        :return: float - 准确率
        """

        if threshold is None:
            if self.threshold is None:
                threshold = 0.5
            else:
                threshold = self.threshold


        self.y_pred = np.where(self.sigmoid >= threshold, 1, 0)

        diff = self.y_pred - self.y_true
        true = np.sum(diff == 0)

        return true / self.n_samples

    def F1(self):
        """
        :description: 求分类模型的F1值   2 * precision * recall / (precision + recall)
        :return: float - F1值
        """

        precision = self.precision()
        recall = self.recall()

        return 2 * precision * recall / (precision + recall)

    def ROC(self):
        """
        :description: 绘制ROC曲线
        :return: list - 返回ROC曲线的横纵坐标 TPR, FPR
        """

        TPR, FPR = [], []
        for threshold in np.linspace(0, 1, 200):

            self.y_pred = np.where(self.sigmoid >= threshold, 1, 0)

            same_index = self.y_pred == self.y_true

            TP = np.sum(self.y_true[same_index] == 1)
            FP = np.sum(self.y_pred[~same_index] == 1)
            TN = np.sum(self.y_true[same_index] == 0)
            FN = np.sum(self.y_pred[~same_index] == 0)

            TPR.append(TP / (TP + FN))
            FPR.append(FP / (FP + TN))

        return TPR, FPR


