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

# 입력된 상의 혹은 하의와 어울리는 하의 혹은 상의의 색을 추천
# target: 추천 대상 고객의 history DB
# most: 추천 대상 고객과 상관관계가 높은 고객의 history DB (하나의 DF로 합쳐진 것)
# least: 추천 대상 고객과 상관관계가 낮은 고객의 history DB
# most_cus_id: 추천 대상 고객과 상관관계가 높은 고객의 id list
# least_cus_id: 추천 대상 고객과 상관관계가 낮은 고객의 id list
# most_weight: 추천 대상 고객과 상관관계가 높은 고객의 correlation list
# least_weight: 추천 대상 고객과 상관관계가 낮은 고객의 correlation list
# chosen: 사용자가 고른 상의 혹은 하의의 색 label
# topObottom: 사용자가 고른 옷이 상의인 경우 0, 하의인 경우 1 입력
# return: 추찬할 상의 혹은 하의의 색 label int 1개
def recomendationTOB(target, most, least, most_cus_id, least_cus_id, most_weight, least_weight, chosen, topObottom):
    if topObottom == 0:  # 사용자가 고른 옷이 상의인 경우(하의를 추천받는 경우)
        target = target[target['top_cls'] == chosen]
        most = most[most['top_cls'] == chosen]
        least = least[least['top_cls'] == chosen]

        target.drop(['top_cls'], axis=1, inplace=True)
        most.drop(['top_cls'], axis=1, inplace=True)
        least.drop(['top_cls'], axis=1, inplace=True)

    else:  # 사용자가 고른 옷이 하의인 경우(상의를 추천받는 경우)
        target = target[target['bottom_cls'] == chosen]
        most = most[most['bottom_cls'] == chosen]
        least = least[least['bottom_cls'] == chosen]

        target.drop(['bottom_cls'], axis=1, inplace=True)
        most.drop(['bottom_cls'], axis=1, inplace=True)
        least.drop(['bottom_cls'], axis=1, inplace=True)

    len_most = len(most_cus_id)

    if len_most == 1:
        # 유사한 5의 '타 고객' id별로 DF를 나눈다
        most1 = most[most['cus_id'] == most_cus_id[0]]
        if topObottom == 0:
            most1_list = list(most1.bottom_cls.to_list())
        else:
            most1_list = list(most1.top_cls.to_list())

        most1_batch = []
        for i in range(len(most1) - 5):
            most1_buf = []
            for j in range(5):
                most1_buf.append(most1_list[i + j])
            most1_batch.append(most1_buf)

        most_batch_list = []
        most_batch_list.append(most1_batch)

        most1_x = []
        most1_y = []
        for i in range(len(most1_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most1_batch[i][j])
            most1_x.append(x_buf)
            y_buf.append(most1_batch[i][4])
            most1_y.append(y_buf)

        most1_x = pd.DataFrame(most1_x)
        most1_y = pd.DataFrame(most1_y)
        most_x_list = []
        most_x_list.append(most1_x)
        most_y_list = []
        most_y_list.append(most1_y)

    elif len_most == 2:
        most1 = most[most['cus_id'] == most_cus_id[0]]
        most2 = most[most['cus_id'] == most_cus_id[1]]
        if topObottom == 0:
            most1_list = list(most1.bottom_cls.to_list())
            most2_list = list(most2.bottom_cls.to_list())
        else:
            most1_list = list(most1.top_cls.to_list())
            most2_list = list(most2.top_cls.to_list())

        most1_batch = []
        for i in range(len(most1) - 5):
            most1_buf = []
            for j in range(5):
                most1_buf.append(most1_list[i + j])
            most1_batch.append(most1_buf)
        most2_batch = []
        for i in range(len(most2) - 5):
            most2_buf = []
            for j in range(5):
                most2_buf.append(most2_list[i + j])
            most2_batch.append(most2_buf)

        most_batch_list = []
        most_batch_list.append(most1_batch)
        most_batch_list.append(most2_batch)

        most1_x = []
        most1_y = []
        for i in range(len(most1_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most1_batch[i][j])
            most1_x.append(x_buf)
            y_buf.append(most1_batch[i][4])
            most1_y.append(y_buf)
        most2_x = []
        most2_y = []
        for i in range(len(most2_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most2_batch[i][j])
            most2_x.append(x_buf)
            y_buf.append(most2_batch[i][4])
            most2_y.append(y_buf)

        most1_x = pd.DataFrame(most1_x)
        most1_y = pd.DataFrame(most1_y)
        most2_x = pd.DataFrame(most2_x)
        most2_y = pd.DataFrame(most2_y)

        most_x_list = []
        most_x_list.append(most1_x)
        most_x_list.append(most2_x)
        most_y_list = []
        most_y_list.append(most1_y)
        most_y_list.append(most2_y)

    elif len_most == 3:
        most1 = most[most['cus_id'] == most_cus_id[0]]
        most2 = most[most['cus_id'] == most_cus_id[1]]
        most3 = most[most['cus_id'] == most_cus_id[2]]
        if topObottom == 0:
            most1_list = list(most1.bottom_cls.to_list())
            most2_list = list(most2.bottom_cls.to_list())
            most3_list = list(most3.bottom_cls.to_list())
        else:
            most1_list = list(most1.top_cls.to_list())
            most2_list = list(most2.top_cls.to_list())
            most3_list = list(most3.top_cls.to_list())

        most1_batch = []
        for i in range(len(most1) - 5):
            most1_buf = []
            for j in range(5):
                most1_buf.append(most1_list[i + j])
            most1_batch.append(most1_buf)
        most2_batch = []
        for i in range(len(most2) - 5):
            most2_buf = []
            for j in range(5):
                most2_buf.append(most2_list[i + j])
            most2_batch.append(most2_buf)
        most3_batch = []
        for i in range(len(most3) - 5):
            most3_buf = []
            for j in range(5):
                most3_buf.append(most3_list[i + j])
            most3_batch.append(most3_buf)

        most_batch_list = []
        most_batch_list.append(most1_batch)
        most_batch_list.append(most2_batch)
        most_batch_list.append(most3_batch)

        most1_x = []
        most1_y = []
        for i in range(len(most1_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most1_batch[i][j])
            most1_x.append(x_buf)
            y_buf.append(most1_batch[i][4])
            most1_y.append(y_buf)
        most2_x = []
        most2_y = []
        for i in range(len(most2_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most2_batch[i][j])
            most2_x.append(x_buf)
            y_buf.append(most2_batch[i][4])
            most2_y.append(y_buf)
        most3_x = []
        most3_y = []
        for i in range(len(most3_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most3_batch[i][j])
            most3_x.append(x_buf)
            y_buf.append(most3_batch[i][4])
            most3_y.append(y_buf)

        most1_x = pd.DataFrame(most1_x)
        most1_y = pd.DataFrame(most1_y)
        most2_x = pd.DataFrame(most2_x)
        most2_y = pd.DataFrame(most2_y)
        most3_x = pd.DataFrame(most3_x)
        most3_y = pd.DataFrame(most3_y)

        most_x_list = []
        most_x_list.append(most1_x)
        most_x_list.append(most2_x)
        most_x_list.append(most3_x)
        most_y_list = []
        most_y_list.append(most1_y)
        most_y_list.append(most2_y)
        most_y_list.append(most3_y)

    elif len_most == 4:
        most1 = most[most['cus_id'] == most_cus_id[0]]
        most2 = most[most['cus_id'] == most_cus_id[1]]
        most3 = most[most['cus_id'] == most_cus_id[2]]
        most4 = most[most['cus_id'] == most_cus_id[3]]
        if topObottom == 0:
            most1_list = list(most1.bottom_cls.to_list())
            most2_list = list(most2.bottom_cls.to_list())
            most3_list = list(most3.bottom_cls.to_list())
            most4_list = list(most4.bottom_cls.to_list())
        else:
            most1_list = list(most1.top_cls.to_list())
            most2_list = list(most2.top_cls.to_list())
            most3_list = list(most3.top_cls.to_list())
            most4_list = list(most4.top_cls.to_list())

        most1_batch = []
        for i in range(len(most1) - 5):
            most1_buf = []
            for j in range(5):
                most1_buf.append(most1_list[i + j])
            most1_batch.append(most1_buf)
        most2_batch = []
        for i in range(len(most2) - 5):
            most2_buf = []
            for j in range(5):
                most2_buf.append(most2_list[i + j])
            most2_batch.append(most2_buf)
        most3_batch = []
        for i in range(len(most3) - 5):
            most3_buf = []
            for j in range(5):
                most3_buf.append(most3_list[i + j])
            most3_batch.append(most3_buf)
        most4_batch = []
        for i in range(len(most4) - 5):
            most4_buf = []
            for j in range(5):
                most4_buf.append(most4_list[i + j])
            most4_batch.append(most4_buf)

        most_batch_list = []
        most_batch_list.append(most1_batch)
        most_batch_list.append(most2_batch)
        most_batch_list.append(most3_batch)
        most_batch_list.append(most4_batch)

        most1_x = []
        most1_y = []
        for i in range(len(most1_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most1_batch[i][j])
            most1_x.append(x_buf)
            y_buf.append(most1_batch[i][4])
            most1_y.append(y_buf)
        most2_x = []
        most2_y = []
        for i in range(len(most2_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most2_batch[i][j])
            most2_x.append(x_buf)
            y_buf.append(most2_batch[i][4])
            most2_y.append(y_buf)
        most3_x = []
        most3_y = []
        for i in range(len(most3_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most3_batch[i][j])
            most3_x.append(x_buf)
            y_buf.append(most3_batch[i][4])
            most3_y.append(y_buf)
        most4_x = []
        most4_y = []
        for i in range(len(most4_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most4_batch[i][j])
            most4_x.append(x_buf)
            y_buf.append(most4_batch[i][4])
            most4_y.append(y_buf)

        most1_x = pd.DataFrame(most1_x)
        most1_y = pd.DataFrame(most1_y)
        most2_x = pd.DataFrame(most2_x)
        most2_y = pd.DataFrame(most2_y)
        most3_x = pd.DataFrame(most3_x)
        most3_y = pd.DataFrame(most3_y)
        most4_x = pd.DataFrame(most4_x)
        most4_y = pd.DataFrame(most4_y)

        most_x_list = []
        most_x_list.append(most1_x)
        most_x_list.append(most2_x)
        most_x_list.append(most3_x)
        most_x_list.append(most4_x)
        most_y_list = []
        most_y_list.append(most1_y)
        most_y_list.append(most2_y)
        most_y_list.append(most3_y)
        most_y_list.append(most4_y)

    else:
        most1 = most[most['cus_id'] == most_cus_id[0]]
        most2 = most[most['cus_id'] == most_cus_id[1]]
        most3 = most[most['cus_id'] == most_cus_id[2]]
        most4 = most[most['cus_id'] == most_cus_id[3]]
        most5 = most[most['cus_id'] == most_cus_id[4]]
        if topObottom == 0:
            most1_list = list(most1.bottom_cls.to_list())
            most2_list = list(most2.bottom_cls.to_list())
            most3_list = list(most3.bottom_cls.to_list())
            most4_list = list(most4.bottom_cls.to_list())
            most5_list = list(most5.bottom_cls.to_list())
        else:
            most1_list = list(most1.top_cls.to_list())
            most2_list = list(most2.top_cls.to_list())
            most3_list = list(most3.top_cls.to_list())
            most4_list = list(most4.top_cls.to_list())
            most5_list = list(most5.top_cls.to_list())

        most1_batch = []
        for i in range(len(most1) - 5):
            most1_buf = []
            for j in range(5):
                most1_buf.append(most1_list[i + j])
            most1_batch.append(most1_buf)
        most2_batch = []
        for i in range(len(most2) - 5):
            most2_buf = []
            for j in range(5):
                most2_buf.append(most2_list[i + j])
            most2_batch.append(most2_buf)
        most3_batch = []
        for i in range(len(most3) - 5):
            most3_buf = []
            for j in range(5):
                most3_buf.append(most3_list[i + j])
            most3_batch.append(most3_buf)
        most4_batch = []
        for i in range(len(most4) - 5):
            most4_buf = []
            for j in range(5):
                most4_buf.append(most4_list[i + j])
            most4_batch.append(most4_buf)
        most5_batch = []
        for i in range(len(most5) - 5):
            most5_buf = []
            for j in range(5):
                most5_buf.append(most5_list[i + j])
            most5_batch.append(most5_buf)

        most_batch_list = []
        most_batch_list.append(most1_batch)
        most_batch_list.append(most2_batch)
        most_batch_list.append(most3_batch)
        most_batch_list.append(most4_batch)
        most_batch_list.append(most5_batch)

        most1_x = []
        most1_y = []
        for i in range(len(most1_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most1_batch[i][j])
            most1_x.append(x_buf)
            y_buf.append(most1_batch[i][4])
            most1_y.append(y_buf)
        most2_x = []
        most2_y = []
        for i in range(len(most2_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most2_batch[i][j])
            most2_x.append(x_buf)
            y_buf.append(most2_batch[i][4])
            most2_y.append(y_buf)
        most3_x = []
        most3_y = []
        for i in range(len(most3_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most3_batch[i][j])
            most3_x.append(x_buf)
            y_buf.append(most3_batch[i][4])
            most3_y.append(y_buf)
        most4_x = []
        most4_y = []
        for i in range(len(most4_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most4_batch[i][j])
            most4_x.append(x_buf)
            y_buf.append(most4_batch[i][4])
            most4_y.append(y_buf)
        most5_x = []
        most5_y = []
        for i in range(len(most5_batch)):
            x_buf = []
            y_buf = []
            for j in range(4):
                x_buf.append(most5_batch[i][j])
            most5_x.append(x_buf)
            y_buf.append(most5_batch[i][4])
            most5_y.append(y_buf)

        most1_x = pd.DataFrame(most1_x)
        most1_y = pd.DataFrame(most1_y)
        most2_x = pd.DataFrame(most2_x)
        most2_y = pd.DataFrame(most2_y)
        most3_x = pd.DataFrame(most3_x)
        most3_y = pd.DataFrame(most3_y)
        most4_x = pd.DataFrame(most4_x)
        most4_y = pd.DataFrame(most4_y)
        most5_x = pd.DataFrame(most5_x)
        most5_y = pd.DataFrame(most5_y)

        # most의 RNN은 for문이르므로 list에 넣는다
        most_x_list = []
        most_x_list.append(most1_x)
        most_x_list.append(most2_x)
        most_x_list.append(most3_x)
        most_x_list.append(most4_x)
        most_x_list.append(most5_x)
        most_y_list = []
        most_y_list.append(most1_y)
        most_y_list.append(most2_y)
        most_y_list.append(most3_y)
        most_y_list.append(most4_y)
        most_y_list.append(most5_y)
    
    # 학습할 데이터를 list로 만든다
    if topObottom == 0:
        target_list = list(target.bottom_cls.to_list())
        least_list = list(least.bottom_cls.to_list())
    else:
        target_list = list(target.top_cls.to_list())
        least_list = list(least.top_cls.to_list())

    # minibatch를 만든다(size=5)
    target_batch = []
    for i in range(len(target) - 5):
        target_buf = []
        for j in range(5):
            target_buf.append(target_list[i + j])
        target_batch.append(target_buf)
    least_batch = []
    for i in range(len(least) - 5):
        least_buf = []
        for j in range(5):
            least_buf.append(least_list[i + j])
        least_batch.append(least_buf)

    # input(x)과 label(y)로 나눔
    target_x = []
    target_y = []
    for i in range(len(target_batch)):
        x_buf = []
        y_buf = []
        for j in range(4):
            x_buf.append(target_batch[i][j])
        target_x.append(x_buf)
        y_buf.append(target_batch[i][4])
        target_y.append(y_buf)

    least_x = []
    least_y = []
    for i in range(len(least_batch)):
        x_buf = []
        y_buf = []
        for j in range(4):
            x_buf.append(least_batch[i][j])
        least_x.append(x_buf)
        y_buf.append(least_batch[i][4])
        least_y.append(y_buf)

    #input과 label을 dataframe으로 만듦
    target_x = pd.DataFrame(target_x)
    target_y = pd.DataFrame(target_y)
    least_x = pd.DataFrame(least_x)
    least_y = pd.DataFrame(least_y)

    # target의 GRU모델
    input = Input(shape=(4, ))
    embedding = Embedding(len(target) + 10, 8)
    hidden = embedding(input)
    rnn_1 = GRU(128, return_sequences=False, return_state=True)
    hidden, fw_h = rnn_1(hidden)
    output = tf.keras.layers.Dense(13, activation=tf.nn.softmax)(hidden)
    model_target = tf.keras.Model(inputs=input, outputs=output)
    model_target.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history_target = model_target.fit(target_x, target_y, epochs=100, verbose=1)

    #가장 최근에 입은 4개 색
    target_pred = []
    for i in range(4):
        target_pred.append(target_batch[-1][i + 1])
    target_pred = pd.DataFrame([target_pred])
    predict_target = model_target.predict(target_pred)
    # weight를 0~1사이로 정규화
    def minmax_calculate(arr):
        return (arr - arr.min(axis=1)) / (arr.max(axis=1) - arr.min(axis=1))
    # target의 선호 색 가중치
    target_MinMax = minmax_calculate(predict_target)[0]

    # most의 GRU 모델
    most_list = []
    for i in range(1, len_most + 1):
        most_list.append('most' + str(i))

    most_result = []
    longest = max(len(most_x_list[0]), len(most_x_list[1]), len(most_x_list[2]), len(most_x_list[3]), len(most_x_list[4]))      # 가장 길이가 긴 DF의 길이

    input = Input(shape=(4, ))
    embedding = Embedding(longest + 10, 8)
    hidden = embedding(input)
    rnn_1 = GRU(128, return_sequences=False, return_state=True)
    hidden, fw_h = rnn_1(hidden)
    output = tf.keras.layers.Dense(13, activation=tf.nn.softmax)(hidden)
    model_most = tf.keras.Model(inputs=input, outputs=output)

    model_most.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    test = []
    for i in range(len(most_list)):
        history_most = model_most.fit(most_x_list[i], most_y_list[i], epochs=100, verbose=1)
        #가장 최근에 입은 4개 색
        most_pred = []
        for j in range(4):
            most_pred.append(most_batch_list[i][-1][j + 1])
        most_pred = pd.DataFrame([most_pred])
        predict_most = model_most.predict(most_pred)
        most_MinMax = minmax_calculate(predict_most)
        most_result.append(most_MinMax[0])

    # least의 GRU 모델
    input = Input(shape=(4, ))
    embedding = Embedding(len(least) + 10, 8)
    hidden = embedding(input)
    rnn_1 = GRU(128, return_sequences=False, return_state=True)
    hidden, fw_h = rnn_1(hidden)
    output = tf.keras.layers.Dense(13, activation=tf.nn.softmax)(hidden)
    model_least = tf.keras.Model(inputs=input, outputs=output)

    model_least.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history_least = model_least.fit(least_x, least_y, epochs=100, verbose=1)

    #가장 최근에 입은 4개 색
    least_pred = []
    for i in range(4):
        least_pred.append(least_batch[-1][i + 1])
    least_pred = pd.DataFrame([least_pred])
    predict_least = model_least.predict(least_pred)
    least_MinMax = minmax_calculate(predict_least)[0]

    # 가중치 계산
    # 상관관계가 높은 '타 고객'의 가중치 계산
    most_weight_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(len(most_list)):
        most_weight_buf = most_result[i] * most_weight[i]
        most_weight_sum += most_weight_buf
    # 상관관계가 낮은 '타 고객'의 가중치 계산
    least_weight_sum = least_MinMax * least_weight

    # 전체 가중치 결합 후 가장 가중치가 높은 값을 가진 색을 추천
    return np.argmax(target_MinMax + most_weight_sum + least_weight_sum)