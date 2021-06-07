import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# 추천받은 옷의 색을 기반으로 옷장에 있는 옷 중 한 벌 추천받기
# closet: 추천 대상 고객의 옷장 DB
# resColor: 추천받은 색의  label
# style: 사용자가 처음에 선택한 상/하의으 스타일의 label. 랜덤추천일 경우 사용자가 선호하는 스타일의 label
# topObottom: 사용자가 고른 옷이 상의이면 0, 하의이면 1 입력. 랜덤추천의 경우 상의를 추천받을 때 1, 하의를 추천받을 때 0 입력
# return: 추천하는 상/하의의 옷의 id list. 개수가 여러개의 경우 랜덤으로 한 벌 추천한다
def recClothesSelect(closet, recColor, style, topObottom):
    # rating이 없는 row는 drop한다
    rated = closet.dropna()
    train, test = train_test_split(rated, test_size = 0.1)

    # 각 column의 데이터 개수 파악
    clothes_num = len(closet)
    id_num = len(closet.id.unique())
    type_num = len(closet.type.unique())
    color_num = len(closet.color.unique())
    weather_num = len(closet.weather.unique())
    sort_num = len(closet.sort.unique())

    # DNN 설계
    # input layer
    id_input = Input(shape=(1, ), name='id_input_layer')
    type_input = Input(shape=(1, ), name='type_input_layer')
    weather_input = Input(shape=(1, ), name='weather_input_layer')
    color_input = Input(shape=(1, ), name='color_input_layer')
    sort_input = Input(shape=(1, ), name='sort_input_layer')

    # embedding layer
    id_embedding_layer = Embedding(id_num + 50, 8, name='id_embedding_layer')
    type_embedding_layer = Embedding(type_num + 50, 8, name='type_embedding_layer')
    weather_embedding_layer = Embedding(weather_num + 50, 8, name='weather_embedding_layer')
    color_embedding_layer = Embedding(color_num + 50, 8, name='color_embedding_layer')
    sort_embedding_layer = Embedding(sort_num + 50, 8, name='sort_embedding_layer')

    # vector layer
    id_vector_layer = Flatten(name='id_vector_layer')
    type_vector_layer = Flatten(name='type_vector_layer')
    weather_vector_layer = Flatten(name='weather_vector_layer')
    color_vector_layer = Flatten(name='color_vector_layer')
    sort_vector_layer = Flatten(name='sort_vector_layer')

    concate_layer = Concatenate()

    dense_layer1 = Dense(128, activation='relu')
    dense_layer2 = Dense(32, activation='relu')

    result_layer = Dense(1)

    id_embedding = id_embedding_layer(id_input)
    type_embedding = type_embedding_layer(type_input)
    weather_embedding = weather_embedding_layer(weather_input)
    color_embedding = color_embedding_layer(color_input)
    sort_embedding = sort_embedding_layer(sort_input)

    id_vector = id_vector_layer(id_embedding)
    type_vector = type_vector_layer(type_embedding)
    weather_vector = weather_vector_layer(weather_embedding)
    color_vector = color_vector_layer(color_embedding)
    sort_vector = sort_vector_layer(sort_embedding)

    concat = concate_layer([id_vector, type_vector, weather_vector, color_vector, sort_vector])
    dense1 = dense_layer1(concat)
    dense2 = dense_layer2(dense1)

    result = result_layer(dense2)

    model = Model(inputs=[id_input, type_input, weather_input, color_input, sort_input], outputs=result)
    model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])

    # 모델 학습
    history = model.fit([train.id, train.type, train.weather, train.color, train.sort], train.rating, epochs=50, verbose=1)

    # rating이 없는 옷 모으기
    null_clothes = closet[closet.isnull().any(axis=1)]
    
    # null인 rating 예측하기
    predictions = model.predict([null_clothes.id, null_clothes.type, null_clothes.weather, null_clothes.color, null_clothes.sort])
    new_rating = []
    for p in predictions:
        num = round(p[0], 2)
        new_rating.append(num)

    # 기존 null인 row 삭제하고 예측값 list를 새 row로 추가
    null_clothes.drop('rating', axis=1, inplace=True)
    null_clothes['rating'] = new_rating

    # 기존 rating된 옷들과 rating을 예측한 옷들의 DF를 합친다
    closet_concat = pd.concat([rated, null_clothes])

    if topObottom == 0:     # 사용자가 상의를 선책한 경우(하의를 추천받고 싶은 경우)
        buf1 = closet_concat[closet_concat['type'] == 1]
        buf2 = closet_concat[closet_concat['type'] == 4]
        closet_concat = pd.concat([buf1, buf2])
    else:       # 사용자가 하의를 선택한 경우 (하의를 선택받고 싶은 경우)
        buf1 = closet_concat[closet_concat['type'] == 2]
        buf2 = closet_concat[closet_concat['type'] == 3]
        buf3 = closet_concat[closet_concat['type'] == 5]
        closet_concat = pd.concat([buf1, buf2, buf3])

    # 추천받은 색과 동일한 색의 옷을 추출
    closet_concat = closet_concat[closet_concat['color'] == recColor]

    # 사용자가 선택한 상/하의와 동일한 스타일의 하/상의 추출
    closet_concat = closet_concat[closet_concat['style'] == style]

    # 해당 스타일의 옷이 옷장에 없다면 비슷한 스타일의 옷을 추천받는다.
    if len(closet_concat) == 0:
        closet_concat = pd.concat([closet_concat[closet_concat['style'] == style-1], closet_concat[closet_concat['style'] == style+1]])

    max_rating = max(closet_concat.rating)
    recomended_clothes_id = list(closet_concat[closet_concat['rating'] == max_rating]['id'])

    return recomended_clothes_id