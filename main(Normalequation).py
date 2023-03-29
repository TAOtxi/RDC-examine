from algorithm.Normalequation import Normalequation
import pandas as pd
from tool.preprocessing import Preprocessing
from tool.predict import Evaluation

data = pd.read_csv('datasets/insurance.csv')
prep = Preprocessing(data.values)
prep.zscore([0, 2])
prep.ZeroOneEncoder([1, 4, 5])

for i in range(10):

    X_train, y_train, X_test, y_test = prep.split(frac=0.7)
    NE = Normalequation()
    NE.fit(X_train, y_train)
    pred = Evaluation(X_test, NE.theta, y_test)

    print(pred.R2(), pred.MSE())

# 其中一次输出
"""
0.6408228893222223 38585563.68249268
0.6454361088170748 35591358.18669301
0.6508887979524868 41974206.80340199
0.6554452241304656 35773612.58242971
0.6105713343834187 42686625.26767933
0.6857829647035515 34993521.77270254
0.6258579928302166 41119502.02787476
0.6455901104731645 41091263.258861504
0.6980812526291433 31684121.64161478
0.7002378457909915 30831398.832938798
"""
