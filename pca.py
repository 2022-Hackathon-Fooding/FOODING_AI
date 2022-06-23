from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import data_reader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

dr = data_reader.DataReader()

columns = ['age_10대','age_20대','age_30대','age_40대', 'age_50대 이상',
           'sex_F','sex_M','time_launch','time_dinner','time_late night snack','feeling_plearsure / joy',
           'feeling_sadness / depressed','feeling_annoying','weather_sunny','weather_cloudy / rain','weather_etc']

df = pd.DataFrame(dr.train_X, columns=columns)
df['target']=dr.train_Y

print(df.head(3))

dr_scaled = StandardScaler().fit_transform(df.iloc[:, :-1])

pca = PCA(n_components=2)
pca.fit(dr_scaled)
dr_pca = pca.transform(dr_scaled)
print(dr_pca.shape)

pca_columns=['pca_component_1','pca_component_2']

dr_pca = pd.DataFrame(dr_pca, columns=pca_columns)
dr_pca['target']=dr.train_Y
print(dr_pca.head(3))

markers=['^', 's', 'o']

for i, marker in enumerate(markers):
    x_axis_data = dr_pca[dr_pca['target']==i]['pca_component_1']
    y_axis_data = dr_pca[dr_pca['target']==i]['pca_component_2']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=dr.train_Y[i])

plt.legend()
plt.xlabel('pca_component_1')
plt.ylabel('pca_component_2')

plt.show()

rcf = RandomForestClassifier(random_state=156)
scores = cross_val_score(rcf, dr.train_X, dr.train_Y,scoring='accuracy',cv=3)
print('원본 데이터 교차 검증 개별 정확도:',scores)
print('원본 데이터 평균 정확도:', np.mean(scores))

pca_X = dr_pca[['pca_component_1', 'pca_component_2']]
scores_pca = cross_val_score(rcf, pca_X, dr.train_Y, scoring='accuracy', cv=3 )
print('PCA 변환 데이터 교차 검증 개별 정확도:',scores_pca)
print('PCA 변환 데이터 평균 정확도:', np.mean(scores_pca))



