import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

filename = '\data\origianl\originData.csv'


#pandas read_csv로 불러오기
df = pd.read_csv(filename, encoding='cp949')

X = df.drop('taste', axis=1)
y = df['taste']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=777)

print("Size of X_train: {}".format(X_train.shape))

print("Size of X_test: {}".format(X_test.shape))



df.head()

#labels = ['매콤한','달콤한','짭짤한']
#s = pd.Series(np.random.randn(3),index=labels)
#print(s)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

print(list(X_train.columns))
X_train.head()

X_train.to_csv("data\preprocessing\X_train.csv")
X_test.to_csv("data\preprocessing\X_test.csv")
y_train.to_csv("data\preprocessing\y_train.csv")
y_test.to_csv("data\preprocessing\y_test.csv")