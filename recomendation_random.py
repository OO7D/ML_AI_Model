import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

# 상의와 하의를 동시에 랜덤으로 추천해준다.
# combination: 추천해 줄 고객의 착장 history DB DataFrame
def recomendationRandom(combination):
    # combination = pd.read_csv(target)

    top_list = list(combination.top_cls.to_list())
    bottom_list = list(combination.bottom_cls.to_list())

    #상의를 5개씩 나눔
    top_comb = []
    for i in range(len(top_list) - 5):
        top_buf = []
        for j in range(5):
            top_buf.append(top_list[i + j])
        top_comb.append(top_buf)

    #상의를 input(x)과 label(y)로 나눔
    top_x = []
    top_y = []
    for i in range(len(top_comb)):
        x_buf = []
        y_buf = []
        for j in range(4):
            x_buf.append(top_comb[i][j])
        top_x.append(x_buf)

        y_buf.append(top_comb[i][4])
        top_y.append(y_buf)

    #input과 label을 dataframe으로 만듦
    top_x = pd.DataFrame(top_x)
    top_y = pd.DataFrame(top_y)

    #하의를 5개씩 나눔
    bottom_comb = []
    for i in range(len(bottom_list) - 5):
        bottom_buf = []
        for j in range(5):
            bottom_buf.append(bottom_list[i + j])
        bottom_comb.append(bottom_buf)

    #하의를 input(x)과 label(y)로 나눔
    bottom_x = []
    bottom_y = []

    for i in range(len(bottom_comb)):
        x_buf = []
        y_buf = []
        for j in range(4):
            x_buf.append(bottom_comb[i][j])
        bottom_x.append(x_buf)

        y_buf.append(bottom_comb[i][4])
        bottom_y.append(y_buf)

    #input과 label을 dataframe으로 만듦
    bottom_x = pd.DataFrame(bottom_x)
    bottom_y = pd.DataFrame(bottom_y)

    # GRU
    input = Input(shape=(4, ))
    embedding = Embedding(len(top_comb) + 1, 8)
    hidden = embedding(input)

    rnn_1 = GRU(128, return_sequences=False, return_state=True)
    hidden, fw_h = rnn_1(hidden)

    output = tf.keras.layers.Dense(13, activation=tf.nn.softmax)(hidden)
    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 상의 학습
    history_top = model.fit(top_x, top_y, epochs=100, verbose=1)

    # 가장 최근에 입은 4개 상의
    top_pred = []
    for i in range(4):
        top_pred.append(top_comb[-1][i + 1])
    top_pred = pd.DataFrame([top_pred])

    # 상의 예측
    predict_top = model.predict(top_pred)
    result_top = np.argmax(predict_top, axis=1)

    # 하의 학습
    history_bottom = model.fit(bottom_x, bottom_y, epochs=100, verbose=1)

    # 가장 최근에 입은 4개 하의
    bottom_pred = []
    for i in range(4):
        bottom_pred.append(bottom_comb[-1][i + 1])
    bottom_pred = pd.DataFrame([bottom_pred])

    # 하의 예측
    predict_bottom = model.predict(bottom_pred)
    result_bottom = np.argmax(predict_bottom, axis=1)

    # 색 label 숫자로 받고 싶으면 삭제
    # color = ['red', 'orange', 'yellow', 'green', 'lime', 'blue', 'sky', 'pink', 'purple', 'brown', 'white', 'grey', 'black']
    # 한글로 받고 싶은 경우
    # color = ['빨강', '주황', '노랑', '초록', '연두', '파랑', '하늘', '분홍', '보라', '갈색', '하양', '회색', '검정']

    # 색 이름으로 받고싶은 경우
    # return color[result_top[0]], color[result_bottom[0]]
    # label 숫자로 받고 싶은 경우
    return result_top[0], result_bottom[0]