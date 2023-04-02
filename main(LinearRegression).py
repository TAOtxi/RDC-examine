import pandas as pd
from tool.preprocessing import Preprocessing
from tool.predict import Evaluation
from algorithm.LinearRegression import MiniBGD
from algorithm.Normalequation import Normalequation

data = pd.read_csv('datasets/insurance.csv')
data = data.values
prep = Preprocessing()

data = prep.zscore(data, [0, 2])
data = prep.OneHotEncoder(data, [1, 4, 5])

X_train, y_train, X_test, y_test = prep.split(data)

# 梯度下降模型
LR = MiniBGD(alpha=0.4, beta=1, tol=1e-4)
LR.fit(X_train, y_train)
pred_grad = Evaluation(X_test, LR.theta, y_test)

# 最小二乘法模型
NE = Normalequation()
NE.fit(X_train, y_train)
pred_NE = Evaluation(X_test, NE.theta, y_test)

print('-'*15 + '梯度下降' + '-'*15)
print('R2值%f， 平均方差%f' % (pred_grad.R2(), pred_grad.MSE()))
print('-'*15 + '最小二乘法' + '-'*15)
print('R2值%f， 平均方差%f' % (pred_NE.R2(), pred_NE.MSE()))

