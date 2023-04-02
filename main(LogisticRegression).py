from tool.preprocessing import Preprocessing
from tool.predict import Evaluation
import pandas as pd
import matplotlib.pyplot as plt
from algorithm.LogisticRegression import LogisticRegression
plt.rcParams["font.sans-serif"] = ["SimHei"]

data = pd.read_csv('datasets/gender_classification.csv')
prep = Preprocessing()
data = data.values
data = prep.LabelEncoder(data, [-1])
data = prep.minmax(data, [1, 2])
X_train, y_train, X_test, y_test = prep.split(data)

LG = LogisticRegression(alpha=0.5)
LG.fit(X_train, y_train)
pred = Evaluation(X_test, LG.theta, y_test, threshold=0.57)

print('-'*10 + '逻辑回归' + '-'*10)
print(f'准确率:\t{pred.accuracy()}\n'
      f'精准率:\t{pred.precision()}\n'
      f'召回率:\t{pred.recall()}\n'
      f'F1值:\t{pred.F1()}')

# 画ROC曲线
fig = plt.figure()
ax = fig.add_subplot()
TPR, FPR = pred.ROC()
ax.plot(FPR, TPR)
ax.set_xlabel('FPR', fontsize=20)
ax.set_ylabel('TPR', fontsize=20)
ax.set_title('ROC曲线', fontsize=30)
plt.show()




