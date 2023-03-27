import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tool.preprocessing import Preprocessing
from tool.predict import Evaluation
from LinearRegression import MiniBGD

data = pd.read_csv('insurance.csv')
prep = Preprocessing(data.values)

# 数据无量纲化处理
prep.zscore([0, 2])
prep.ZeroOneEncoder([1, 4, 5])

# 划分训练集和测试集
X_train, y_train, X_test, y_test = prep.split(frac=0.7)

Z = []
for alpha in np.linspace(0.01, 0.45, 30):

    Z_x = []
    for beta in np.linspace(0.01, 0.45, 30):

        LR = MiniBGD(alpha=alpha, beta=beta, tol=1e-4, maxiter=1e4)
        LR.fit(X_train, y_train)
        Z_x.append(LR.loss[-1][0][0])
        pred = Evaluation(X_test, LR.theta, y_test)

        print('alpha = {}\t\tbeta = {}\t\tloss = {}\t\tR2 = {}'.format(alpha, beta, LR.loss[-1][0][0], pred.R2()))

    Z.append(Z_x)
x = y = np.linspace(0.01, 0.45, 30)
X, Y = np.meshgrid(x, y)
Z = np.array(Z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='rainbow')
plt.show()

