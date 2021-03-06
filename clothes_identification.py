from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

# 입력된 옷 이미지가 어떤 옷인지 판별
# img: 판별할 옷 이미지. Image.open(경로)
# return: 이미지가 어떤 옷인지 label
def clothesIdentify(img):
    image_w = 64
    image_h = 64

    X = []

    img = img.convert("L")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    X.append(data)

    X = np.array(X)
    model = load_model('.model')  # tensorflow CNN 학습모델 경로

    X = X.reshape(-1, 64, 64, 1)

    prediction = model.predict(X)

    # label 숫자값을 받고 싶으면 삭제
    # clothes_att = ['dress', 'pants', 'pullover', 'shirts', 'skirts', 't-shirts']
    # 한국어로 받고싶은 경우
    # clothes_att = ['드레스/원피스', '바지', '니트/맨투맨', '셔츠', '치마', '티셔츠']

    predict_label = prediction.argmax()

    # str값을 받고 싶으면 clothes_att[predict_label] return
    return predict_label
