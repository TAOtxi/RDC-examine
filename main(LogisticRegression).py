from tool.preprocessing import Preprocessing
from tool.predict import Evaluation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from sklearn.linear_model import LogisticRegression as LR

data = pd.read_csv('gender_classification.csv')
prep = Preprocessing(data.values)
prep.LabelEncoder([-1])
X_train, y_train, X_test, y_test = prep.split()

for alpha in np.linspace(0.05, 0.8, 30):

    LG = LogisticRegression(alpha=alpha)

    LG.fit(X_train, y_train)

    pred = Evaluation(X_test, LG.theta, y_test, 0.5)
    print(pred.accuracy())

LR = LR()
LR.fit(X_train, y_train.flatten())

print(LR.score(X_test, y_test))




