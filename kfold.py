from sklearn.linear_model import LinearRegression # 선형 회귀 모델을 불러옵니다.
from sklearn.model_selection import KFold # 데이터를 K개의 폴드로 분할해주는 패키지를 사용하겠습니다.
import data_reader

dr = data_reader.DataReader()
# 예측모델 인스턴스를 만듭니다.
lr = LinearRegression()

# KFold 함수를 이용해 폴드 개수를 할당해줍니다.(임의로 5개의 폴드를 사용합니다. 일반적으로 5,10 등의 값을 사용합니다.)
kfold = KFold(n_splits=5)

cv_rmse = []  # 각 cv회차의 rmse 점수를 계산하여 넣어줄 리스트를 생성합니다. 이후 RMSE값의 평균을 구하기 위해 사용됩니다.
n_iter = 0  # 반복 횟수 값을 초기 설정해줍니다. 이후 프린트문에서 각 교차검증의 회차를 구분하기 위해 사용됩니다.

# K값이 5이므로 이 반복문은 5번 반복하게 됩니다.
for train_index, test_index in kfold.split(dr.train_X):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할합니다. 인덱스 값을 분할해줍니다.

    x_train, x_test = feature.iloc[train_index], feature.iloc[test_index]  # feature로 사용할 값을 나눠진 인덱스값에 따라 설정합니다.
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]  # label로 사용할 값을 나눠진 인덱스값에 따라 설정합니다.

    lr = lr.fit(x_train, y_train)  # 모델 학습
    pred = lr.predict(x_test)  # 테스트셋 예측
    n_iter += 1  # 반복 횟수 1회 증가

    error = RMSE(y_test, pred)  # RMSE 점수를 구합니다.
    train_size = x_train.shape[0]  # 학습 데이터 크기
    test_size = x_test.shape[0]  # 검증 데이터 크기

    print('\n{0}번째 교차 검증 RMSE : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
          .format(n_iter, error, train_size, test_size))
    print('{0}번째 검증 세트 인덱스 : {1}'.format(n_iter, test_index))
    cv_rmse.append(error)

print('\n==> 이 방정식의 평균 에러(RMSE)는 {} 입니다.'.format(np.mean(cv_rmse)))  # 모델의 평균정확도를 확인합니다.