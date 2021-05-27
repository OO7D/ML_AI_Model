from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

import joblib

# 이미지에서 8개의 색의 hex값과 각각의 가중치를 구한다.
# url: 값을 구할 이미지의 경로(str)
# output: 구해진 색의 hex값(list), 색의 가중치(list)
def colorClustering(url):
    img = cv2.imread(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def RGB2HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    # def get_image(image_path):
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     return image

    modified = cv2.resize(img, (600, 400), interpolation = cv2.INTER_AREA)
    modified = modified.reshape(modified.shape[0]*modified.shape[1], 3)

    color_num = 8

    clf = KMeans(n_clusters = color_num)
    labels = clf.fit_predict(modified)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    #rgb_colors = [ordered_colors[i] for i in counts.keys()]

    color_percent = []
    for i in counts.values():
        color_percent.append(i)

    return hex_colors, color_percent

# Random Forest로 구한 값의 색을 판별하고, 가장 흔하게 나타나는 색을 return한다.
# hex_colors: 색의 hex값(list). [#xxxxxx, #xxxxxx, ...]
# color_percent: 색의 가중치(list). [int, int, ...]
# colorClustering 함수 사용 시 출력값 가공 없이 그대로 입력 사용 가능
# output: 판별된 색의 이름 2개(list), 판별된 각 색의 백분율 2개(list)
def colorIdentify(hex_colors, color_percent):
    model = joblib.load('')     # Random forest joblib파일 경로

    # 색을 label 숫자로 받고 싶으면 삭제
    color_label = ["red", "orange", "yellow", "green", "lime", "blue", "sky", "purple", "pink", "brown", "white", "grey", "black"]
    # 한글로 받고 싶은 경우
    # color_label = ["빨강", "주황", "노랑", "초록", "연두", "파랑", "하늘", "보라", "분홍", "갈색", "하양", "회색", "검정"]

    def rgb_value(color_hex):
        color_hex = color_hex.lstrip('#')
        length = len(color_hex)
        return tuple(int(color_hex[i:i + length // 3], 16) for i in range(0, length, length // 3))

    def color_predict(colors):
        color_arr = [] #예측한 색 label
        color_pre = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #예측 가중치 합산
        c_percent = [] #가중치 백분율(%) 합산

        max_color = [] #최고 백분율 색 2개
        max_pre = [] #최고 백분율 값 2개
        second_percent = 0 #두 번째로 큰 백분율 
    
        for i in range(len(colors)):
            color_arr.append(model.predict(np.array([rgb_value(colors[i])]))[0])

        for i in range(len(color_arr)):
            color_pre[color_arr[i]] += color_percent[i]

        weight_sum = sum(color_pre)

        #백분율 구하기
        for i in color_pre:
            c_percent.append(round(i / weight_sum) * 1000)

        max_index = color_pre.index(max(color_pre))
        max_percent = max(c_percent)

        #두 번째로 큰 백분율 구하기
        for i in c_percent:
            if i > max_percent:
                second_percent = max_percent
                max_percent = i
            elif second_percent < i < max_percent:
                second_percent = i

        second_index = c_percent.index(second_percent)

        max_color.append(color_label[max_index])
        max_color.append(color_label[second_index])

        max_pre.append(max_percent)
        max_pre.append(second_percent)
    
        return max_color, max_index

    identified_color, identified_percent = color_predict(hex_colors)

    return identified_color, identified_percent
