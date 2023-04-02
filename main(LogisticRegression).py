from tool.preprocessing import Preprocessing
from tool.predict import Evaluation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from algorithm.LogisticRegression import LogisticRegression
from sklearn.linear_model import LogisticRegression as LR

data = pd.read_csv('datasets/gender_classification.csv')
prep = Preprocessing(data.values)
prep.LabelEncoder([-1])

# prep.zscore([1, 2])
prep.minmax([1, 2])

X_train, y_train, X_test, y_test = prep.split()

max = []
fig, ax = plt.subplots()
for alpha in np.linspace(0.01, 4, 100):

    LG = LogisticRegression(alpha=alpha)
    LG.fit(X_train, y_train)

    pred = Evaluation(X_test, LG.theta, y_test)
    TPR, FPR = pred.ROC()
    ax.plot(FPR, TPR)
    # plt.show()
    for threshold in np.linspace(0., 1., 200):

        max.append([pred.accuracy(threshold), threshold, alpha])
    # plt.plot(FPR, TPR)
    # print(TPR, FPR)
    # plt.title('alpha = {}'.format(alpha))

plt.show()
max = np.array(max)
maxlist = np.argsort(max[:, 0])[::-1][:20]
for i in max[maxlist]:
    print('accuracy = {}\t\tthreshold = {}\t\talpha = {}'.format(i[0], i[1], i[2]))
print("前20的决策边界的平均值", np.mean(max[maxlist][:, 1]))
LR = LR()
LR.fit(X_train, y_train.flatten())

print(LR.score(X_test, y_test))




