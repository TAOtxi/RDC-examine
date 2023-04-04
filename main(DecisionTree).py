from algorithm.DecisionTree import DecisionTree
from tool.preprocessing import Preprocessing
import pandas as pd
import numpy as np


data = pd.read_csv('datasets/gender_classification.csv')
FeatureNames = data.columns[:-1].tolist()

data = np.array(data)

prep = Preprocessing()
train, test = prep.split(data=data, split_XY=False, convert=False)

tree = DecisionTree(train, FeatureNames)
tree.Discret(columns=[1, 2])

tree.fit()
test[:, 1] = np.where(test[:, 1] > tree.best_divide[1], 1, 0)
test[:, 2] = np.where(test[:, 2] > tree.best_divide[2], 1, 0)
print('决策树训练集准确率：', tree.accuracy(train))
print('决策树测试集准确率：', tree.accuracy(test))
print('剪枝前的决策树', tree.tree)
newtree = tree.PostPruning(train, test)
print('剪枝后训练集准确率：', tree.accuracy(train, newtree))
print('剪枝后测试集准确率：', tree.accuracy(test, newtree))
print('剪枝后的决策树', newtree)

# 其中一种输出:
"""
决策树训练集准确率： 0.962
决策树测试集准确率： 0.966022651565623
剪枝前的决策树 {'nose_wide': {0: {'lips_thin': {0: {'distance_nose_to_lip_long': {0: {'nose_long': {0: 'Female', 1: 'Female'}}, 1: {'nose_long': {0: 'Female', 1: {'forehead_width_cm': {0: 'Female', 1: {'forehead_height_cm': {0: 'Male', 1: {'long_hair': {0: 'Female', 1: 'Female'}}}}}}}}}}, 1: {'distance_nose_to_lip_long': {0: {'nose_long': {0: 'Female', 1: {'forehead_height_cm': {0: 'Male', 1: {'forehead_width_cm': {0: 'Female', 1: 'Female'}}}}}}, 1: {'forehead_width_cm': {0: 'Female', 1: {'nose_long': {0: {'long_hair': {0: {'forehead_height_cm': {0: 'Male', 1: 'Male'}}, 1: {'forehead_height_cm': {0: 'Female', 1: 'Male'}}}}, 1: 'Male'}}}}}}}}, 1: {'forehead_width_cm': {0: 'Female', 1: {'nose_long': {0: {'distance_nose_to_lip_long': {0: {'lips_thin': {0: {'long_hair': {0: 'Female', 1: {'forehead_height_cm': {0: 'Female', 1: 'Female'}}}}, 1: {'long_hair': {0: 'Female', 1: 'Female'}}}}, 1: {'lips_thin': {0: {'forehead_height_cm': {0: 'Male', 1: {'long_hair': {0: 'Male', 1: 'Female'}}}}, 1: 'Male'}}}}, 1: {'distance_nose_to_lip_long': {0: {'lips_thin': {0: {'forehead_height_cm': {0: {'long_hair': {0: 'Female', 1: 'Male'}}, 1: {'long_hair': {0: 'Male', 1: 'Male'}}}}, 1: {'forehead_height_cm': {0: {'long_hair': {0: 'Male', 1: 'Male'}}, 1: 'Male'}}}}, 1: {'lips_thin': {0: 'Male', 1: 'Male'}}}}}}}}}}
剪枝后训练集准确率： 0.9582857142857143
剪枝后测试集准确率： 0.9686875416389074
剪枝后的决策树 {'nose_wide': {0: {'lips_thin': {0: {'distance_nose_to_lip_long': {0: {'nose_long': {0: 'Female', 1: 'Female'}}, 1: {'nose_long': {0: 'Female', 1: {'forehead_width_cm': {0: 'Female', 1: {'forehead_height_cm': {0: 'Female', 1: {'long_hair': {0: 'Female', 1: 'Female'}}}}}}}}}}, 1: {'distance_nose_to_lip_long': {0: {'nose_long': {0: 'Female', 1: {'forehead_height_cm': {0: 'Female', 1: {'forehead_width_cm': {0: 'Female', 1: 'Female'}}}}}}, 1: {'forehead_width_cm': {0: 'Female', 1: {'nose_long': {0: {'long_hair': {0: {'forehead_height_cm': {0: 'Male', 1: 'Male'}}, 1: {'forehead_height_cm': {0: 'Female', 1: 'Male'}}}}, 1: 'Male'}}}}}}}}, 1: {'forehead_width_cm': {0: 'Male', 1: {'nose_long': {0: {'distance_nose_to_lip_long': {0: {'lips_thin': {0: {'long_hair': {0: 'Female', 1: {'forehead_height_cm': {0: 'Female', 1: 'Female'}}}}, 1: {'long_hair': {0: 'Female', 1: 'Female'}}}}, 1: {'lips_thin': {0: {'forehead_height_cm': {0: 'Male', 1: {'long_hair': {0: 'Male', 1: 'Male'}}}}, 1: 'Male'}}}}, 1: {'distance_nose_to_lip_long': {0: {'lips_thin': {0: {'forehead_height_cm': {0: {'long_hair': {0: 'Female', 1: 'Male'}}, 1: {'long_hair': {0: 'Male', 1: 'Male'}}}}, 1: {'forehead_height_cm': {0: {'long_hair': {0: 'Male', 1: 'Male'}}, 1: 'Male'}}}}, 1: {'lips_thin': {0: 'Male', 1: 'Male'}}}}}}}}}}
"""
