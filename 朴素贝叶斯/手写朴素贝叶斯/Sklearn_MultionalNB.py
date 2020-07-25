"""
使用sklearn进行处理的标准过程
分析：数据X是特征列6个样本，100个特征
      Y是类别6个类，标签分别是1，2，3，4，5，6
      直接投入特征列和列别列进行数据fit训练
      直接predicate测试数据就能得到输出的结果
"""
"""
The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). 
The multinomial distribution normally requires integer feature counts. 
However, in practice, fractional counts such as tf-idf may also work.
翻译：多项式朴素贝叶斯模型适合于离散型特征，例如基于词数量的文本分类
    这个多项式分布通常情况下需要整数类型的特征
    然而，事实上，像tf-idf这样的特征也是可以起作用的
"""
import numpy as np
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
X_test = rng.randint(5, size=(2, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
print(clf.predict(X_test))
