import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tool.preprocessing import Preprocessing
from tool.predict import Evaluation
from LinearRegression import MiniBGD

data = pd.read_csv('insurance.csv')
prep = Preprocessing(data.values)

prep.zscore([0, 2])
prep.ZeroOneEncoder([1, 4, 5])

X_train, y_train, X_test, y_test = prep.split(frac=0.7)

model_view = []
temp = []
for beta in np.linspace(0.01, 1, 30):

    for alpha in np.linspace(0.01, 0.45, 30):

        LR = MiniBGD(alpha=alpha, beta=beta, tol=1e-4, maxiter=1e4)
        LR.fit(X_train, y_train)
        pred = Evaluation(X_test, LR.theta, y_test)
        temp = [alpha, beta, LR.loss[-1], pred.R2()]
        model_view.append(temp)
model_view = pd.DataFrame(model_view, columns=['alpha', 'beta', 'loss', 'R2'])
print(model_view)
