import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tool.preprocessing import Preprocessing
from tool.predict import Evaluation
from LinearRegression import MiniBGD
from sklearn.linear_model import Ridge

data = pd.read_csv('insurance.csv')
prep = Preprocessing(data)

# prep.minmax(data.bmi, data.age)
prep.zscore(data.bmi, data.age)

# prep.LabelEncoder(data.sex, data.smoker, data.region)
prep.ZeroOneEncoder(data.sex, data.smoker, data.region)

X_train, y_train, X_test, y_test = prep.split(frac=0.7, seed=None)

# a = []
for alpha in np.linspace(0.01, 0.45, 30):
# for alpha in np.linspace(0.00001, 0.0017, 10):

    beta = 0.01
    LR = MiniBGD(alpha=alpha, beta=beta, tol=1e-4, maxiter=1e4)
    LR.fit(X_train, y_train)
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)

    theta = np.insert(ridge.coef_, 0, ridge.intercept_).reshape(-1, 1)
    pred_1 = Evaluation(X_test, LR.theta, y_test)
    pred_2 = Evaluation(X_test, theta, y_test)
    print(pred_1.R2(), '\t\t',
          pred_2.R2(), '\t\t',
          pred_1.R2() - pred_2.R2(), '\t\t',
          pred_1.MSE(), '\t\t',
          pred_2.MSE(), '\t\t',
          pred_1.MSE() - pred_2.MSE())
    # print(pred_1.MAE())
    # with open('rate.txt', 'a') as fb:
    #     fb.write('alpha=%f, beta=%f\n' % (alpha, beta))
    #     fb.write(str(LR.iter))
    #     fb.write('\n{}\t\t{}\t\t{}\n'.format(LR.theta[0][0], ridge.intercept_[0], LR.theta[0][0] - ridge.intercept_[0]))
    #     for i in range(data.shape[1] - 1):
    #         fb.write('{}\t\t{}\t\t{}\n'.format(LR.theta[i + 1][0], ridge.coef_[0][i], LR.theta[i + 1][0] - ridge.coef_[0][i]))
    #     fb.write('loss:\t' + str(LR.loss[-1][0][0]))
    #     fb.write('\n' + '-'*100 + '\n')

    # a.append(np.array(LR.loss).flatten()

# b = 1
# for i in a:
#     plt.subplot(len(a), 1, b)
#     b += 1
#     plt.figure(figsize=(30, 20))
#     plt.plot(np.arange(i.shape[0]), i)
# plt.show()
