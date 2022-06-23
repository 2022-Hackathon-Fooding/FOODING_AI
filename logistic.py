from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
import data_reader
import numpy as np
import pandas as pd

dr = data_reader.DataReader()
print(dr)
lr = LogisticRegression(C=10, max_iter=1000,multi_class='auto')
lr.fit(dr.train_X, dr.train_Y)

print(lr.predict(dr.test_X[:11]))

proba = lr.predict_proba(dr.test_X[:11])
print(np.round(proba, decimals=3))

print(lr.classes_)

print(lr.coef_.shape, lr.intercept_.shape)

decision = lr.decision_function(dr.test_X[:11])
print(np.round(decision, decimals=2))

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))

print(lr.score(dr.train_X, dr.train_Y))
print(lr.score(dr.test_X, dr.test_Y))

print("훈련 세트 정확도: {:.3f}".format(lr.score(dr.train_X, dr.train_Y)))
print("테스트 세트 정확도: {:.3f}".format(lr.score(dr.test_X, dr.test_Y)))