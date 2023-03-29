import numpy as np

class Normalequation:

    def __init__(self, beta=1.0):
        """
        :param beta: float 正则化系数
        """
        self.theta = None
        self.beta = beta

    def fit(self, X, y):
        """
        :param X: np.ndarray 特征矩阵. shape = (n_samples, n_features)
        :param y: np.ndarray 标签. shape = (n_samples, 1)
        :return: None
        """
        X = np.insert(X, 0, 1, axis=1)
        self.theta = np.linalg.inv(X.T @ X + self.beta * np.eye(X.shape[1])) @ X.T @ y
