from algorithm.BayesClassifier import Bayes
from tool.preprocessing import Preprocessing
import pandas as pd

data = pd.read_csv('datasets/gender_classification.csv').values
pred = Preprocessing()
train, test = pred.split(data=data, split_XY=False, convert=False)
bayes = Bayes()

bayes.fit(train, [0, 3, 4, 5, 6], [1, 2])
print("贝叶斯分类器准确率:\n", bayes.accuracy(test[:, :-1], test[:, -1]))
