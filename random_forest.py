import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import data_reader

dr = data_reader.DataReader()

rf = RandomForestClassifier(n_estimators=100)
rf.fit(dr.train_X, dr.train_Y)
pred1 = rf.predict(dr.train_X)
accuracy1 = accuracy_score(dr.train_Y, pred1)
print("훈련 세트 정확도: {:.3f}".format(accuracy1))

pred2 = rf.predict(dr.test_X)
accuracy2 = accuracy_score(dr.test_Y, pred2)
print("테스트 세트 정확도: {:.3f}".format(accuracy2))



