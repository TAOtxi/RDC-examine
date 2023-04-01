# 文件说明

- *.ipynb: 数据预处理的过程和模型参数选择的过程

- main(.*).py: 根据ipynb的结果而选出的模型

- tool: 封装有对模型评估(predict.py)和数据处理(preprocessing.py)的类

- algorithm: 训练模型的几个算法

- datasets: 数据集

# 更新日记

## 4.1晚

被决策树卡几天了，今晚上选修课，看代码看烦了，就写写这个。
先说下之前的提交，我觉得最重要的是那个数据类型的转换。
我之前用的虽然是array类型，不过里面的元素都是float或int型，
或许还要感谢之前的脑抽行为，一开始的提交里，数据处理的部分都是基于
DataFrame类型运算，之后我突然发现了这个毛病，经过好些时间改错后换成了基于array类型
运算。不过在遍历学习率的时候，程序运算时间慢了超多。找了几天问题后，
才知道是这个类型的不同。基于DataFrame那会儿，将Series转成array时，Series.values返回的数组元素类型是np.float64, 而np.array返回的array类型里是float型。。。
有趣的是，当时我问bing的ai为什么np.float64运算那么快，它说np.float64运算是上古黑科技，
哈哈。其它的...也没啥好讲了。。。