"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
import tensorflow as tf
from tensorflow import keras
import data_reader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import tensorflowjs as tfjs
import numpy as np

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 200  # 예제 기본값은 20입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

#csv = pd.read_csv("data/data_after_one-hot-encoding.csv")
#csv2=csv.drop(["spicy"],axis=1)
#print(csv2)

#csv_y = csv[["spicy"]]
#print(csv_y)

#train_X1, test_X1, train_Y1, test_Y1 = train_test_split(csv2, csv_y, test_size=0.2, stratify=csv_y, random_state=50)

# 인공신경망을 제작합니다.
model = keras.Sequential([
    keras.layers.Dense(16),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(32, activation="relu"),
    #keras.layers.Dropout(rate=0.2),
    #keras.layers.Dense(128, activation="relu"),
    #keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(3, activation='softmax')
])

# 인공신경망을 컴파일합니다.
model.compile(optimizer="adam", metrics=["accuracy"],
              loss="sparse_categorical_crossentropy")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode= 'auto') ,callbacks=[early_stop]
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS, batch_size=3,
                    validation_data=(dr.test_X, dr.test_Y)
                    )

model.summary()



#scores = cross_val_score(model, dr.test_X, dr.test_Y, cv=3)

#print('교차 검증별 정확도:',np.round(scores, 4))
#print('평균 검증 정확도:', np.round(np.mean(scores), 4))



#tfjs.converters.save_keras_model(model, "data/")



