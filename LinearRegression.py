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
        self.tol = tol          # 收敛最小容忍度
        self.maxiter = maxiter  # 最大迭代次数
        self.iter = maxiter     # 迭代次数
        self.loss = []          # 每次迭代后的代价组成的列表
        self.theta = None

    # 损失函数
    def lossfunc(self, X, y, theta, beta):

        diff = X @ theta - y
        grad = (X.T @ diff + beta * theta) / X.shape[0]
        self.loss.append((diff.T @ diff + theta.T @ theta) / X.shape[0])
        return grad

    # 梯度下降
    def grad_dec(self, X, y):

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.theta = np.ones((X.shape[1], 1))
        X_random = X.sample(frac=1)
        grad = self.lossfunc(X, y, self.theta, self.beta)

        for iter_num in range(int(self.maxiter)):
            self.theta = self.theta - self.alpha * grad
            grad = self.lossfunc(X, y, self.theta, self.beta)

            if np.abs(self.loss[-2] - self.loss[-1]) < self.tol:
                self.iter = iter_num
                break

