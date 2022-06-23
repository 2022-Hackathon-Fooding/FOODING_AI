from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import data_reader
form skleran.model.selection import GridSearchCV
import pandas as pd


dr = data_reader.DataReader()


tree = DecisionTreeClassifier()

param_grid={'max_depth':[None, 1, 2, 3, 4, 5],
            'max_leaf_nodes':[3, 5, 7, 9]
            }

grid_search = GridSearchCV(tree, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=1)

tree.fit(dr.train_X, dr.train_Y)




print("훈련 세트 정확도: {:.3f}".format(tree.score(dr.train_X, dr.train_Y)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(dr.test_X, dr.test_Y)))

df = pd.DataFrame(grid_search.cv_result)
df[df.colums[6:]].sort_values('rank_test_score').head()