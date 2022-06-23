"""
Author : Byunghyun Ban
Date : 2020.07.24.
"""
#file = open("data/after_one-hot_encoding_new.csv")
import tensorflow as tf
import data_reader
import network

# 학습 결과를 그래프로 출력합니다.

test_d=[[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]]

answer=network.model.predict(test_d)

#answer1=answer[0][0]
#answer2=answer[0][1]
#answer3=answer[0][2]

print(answer)

data_reader.draw_graph(network.history)
