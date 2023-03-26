import numpy as np
import pandas as pd

class MiniBGD:

    def __init__(
            self,
            alpha=1.0,
            beta=1.0,
            tol=1e-3,
            maxiter=1e4
    ):
        self.alpha = alpha      # 学习率
        self.beta = beta        # 正则化系数
        self.tol = tol          # 判断收敛最小容忍度
        self.maxiter = maxiter  # 最大迭代次数
        self.iter = maxiter     # 迭代次数
        self.loss = []          # 每次迭代后的代价组成的列表
        self.theta = None       # 模型参数
        self.ran_num = None     # 随机抽取的样本数

    def lossfunc(self, X, y):
        """
        :param X: 特征矩阵
        :param y: 标签
        :return: 损失函数的梯度
        """

        diff = X @ self.theta - y
        grad = (X.T @ diff + self.beta * self.theta) / self.ran_num
        self.loss.append((diff.T @ diff + self.theta.T @ self.theta) / self.ran_num)
        return grad

    def fit(self, X, y, ran_num=None):
        """
        :param X: 特征矩阵
        :param y: 标签
        :param ran_num: 随机抽取的样本数（默认为全部）
        :return: None
        """

        if ran_num is None:
            self.ran_num = X.shape[0]
        else:
            self.ran_num = ran_num

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.theta = np.ones((X.shape[1], 1))
        X_random, y_random = self.random(X, y)
        grad = self.lossfunc(X_random, y_random)

        for iter_num in range(int(self.maxiter)):
            self.theta = self.theta - self.alpha * grad
            X_random, y_random = self.random(X, y)
            grad = self.lossfunc(X_random, y_random)

            if np.abs(self.loss[-2] - self.loss[-1]) < self.tol:
                self.iter = iter_num
                break

    def random(self, X, y):
        """
        :param X: 特征矩阵
        :param y: 标签
        :return: 随机抽取的样本
        """

        # 如果抽取的样本数等于样本总数，则返回全部样本
        if self.ran_num == X.shape[0]:
            return X, y

        sample_list = np.random.permutation(X.shape[0])[:self.ran_num]
        X_random = X[sample_list]
        y_random = y[sample_list]

        return X_random, y_random
