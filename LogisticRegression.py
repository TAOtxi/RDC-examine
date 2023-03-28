import numpy as np

class LogisticRegression:

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
        self.batch_size = None  # 随机抽取的样本数

    def sigmoid(self, z):
        """
        :description: sigmoid函数
        :param z: np.ndarray. shape = (batch_size, 1)
            z = X @ theta.
        :return: np.ndarray.
        """
        return 1 / (1 + np.exp(-z))

    def lossfunc(self, X, y):
        """
        :description: 计算损失函数的梯度
        :param X: np.ndarray 特征矩阵. shape = (batch_size, n_features)
        :param y: np.ndarray 标签. shape = (batch_size, 1)
        :return: 损失函数的梯度
        """
        sig = self.sigmoid(X @ self.theta)
        self.loss.append((-y.T @ np.log(sig) - (1 - y).T @ np.log(1 - sig) + self.theta.T @ self.theta)[0][0] / self.batch_size)
        grad = (X.T @ (sig - y) + self.beta * self.theta) / self.batch_size

        return grad

    def fit(self, X, y, batch_size=None):
        """
        :description: 拟合模型
        :param X: np.ndarray 特征矩阵. shape = (n_samples, n_features)
        :param y: np.ndarray 标签. shape = (n_samples, 1)
        :param batch_size: int 随机抽取的样本数（默认为全部）
        :return: None
        """

        if batch_size is None:
            self.batch_size = X.shape[0]
        else:
            self.batch_size = batch_size

        X = np.insert(X, 0, 1, axis=1)
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
        :description: 随机抽取样本
        :param X: np.ndarray 特征矩阵. shape = (n_samples, n_features)
        :param y: np.ndarray 标签. shape = (n_samples, 1)
        :return: 随机抽取的样本
        """

        # 如果抽取的样本数等于样本总数，则返回全部样本
        if self.batch_size == X.shape[0]:
            return X, y

        sample_index = np.random.randint(0, X.shape[0], self.batch_size)
        X_random = X[sample_index]
        y_random = y[sample_index]

        return X_random, y_random


