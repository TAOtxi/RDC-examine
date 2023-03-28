import numpy as np

class Evaluation:

    def __init__(self, X, theta, y_true, threshold=None):
        """
        :description: 评估模型. (MAE, MSE, RMSE, R2, precision, recall, accuracy)
        :param X: pd.DataFrame.     shape = (n_samples, n_features)
        :param theta: np.array.     shape = (n_features, 1)
        :param y_true: np.array.    shape = (n_samples, 1)
        :param threshold: float.    分类阈值
        """
        self.y_true = y_true
        self.y_pred = np.insert(X, 0, 1, axis=1) @ theta
        self.n_samples = y_true.shape[0]
        self.threshold = threshold

        if self.threshold is not None:
            self.y_pred = np.where(self.y_pred > self.threshold, 1, 0)

    def MAE(self):
        """
        :description: Mean Absolute Error
        :return: MAE
        """
        return np.mean(np.abs(self.y_pred - self.y_true)) / self.n_samples

    def MSE(self):
        """
        :description: Mean Squared Error
        :return: MSE
        """
        diff = self.y_pred - self.y_true

        # diff.T @ diff.shape ==> (1, 1)
        return (diff.T @ diff / self.n_samples)[0][0]

    def RMSE(self):
        """
        :description: Root Mean Squared
        :return: RMSE
        """
        return np.sqrt(self.MSE())

    def R2(self):
        """
        :description: R2 score (coefficient of determination)
        :return: R2
        """
        y_mean = np.mean(self.y_true)
        diff_true = self.y_pred - self.y_true
        diff_mean = self.y_pred - y_mean
        return 1 - ((diff_true.T @ diff_true) / (diff_mean.T @ diff_mean))[0][0]

    def precision(self):
        """
        :description: precision score TP / (TP + FP)
            分类模型的精确率
        :return: precision
        """

        diff = self.y_pred - self.y_true
        TP = np.sum(diff == 0)
        FP = np.sum(diff == 1)

        return TP / (TP + FP)

    def recall(self):
        """
        :description: recall score TP / (TP + FN)
            分类模型的召回率
        :return: recall
        """

        diff = self.y_pred - self.y_true
        TP = np.sum(diff == 0)
        FN = np.sum(diff == -1)

        return TP / (TP + FN)

    def accuracy(self):
        """
        :description: precision score (TP+TN)/(TP+TN+FP+FN)
            分类模型的准确率
        :return: accuracy
        """
        self.y_pred = np.where(self.y_pred >= self.threshold, 1, 0)
        diff = self.y_pred - self.y_true
        TP = np.sum(diff == 0)

        return TP / self.n_samples

    def F1(self):
        """
        :description: F1 score 2 * precision * recall / (precision + recall)
            分类模型的F1值
        :return: F1
        """
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

