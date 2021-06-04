import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# 이미지의 윤곽선을 판별해 opbject에 딱 맞는 크기로 이미지를 자른다.
# img: 이미지 cv2.imread(경로)
# img_grey: 흑백 이미지 cv2.imread(경로, cv2.IMREAD_GRAYSCALE)
# 출력: 여백을 자른 이미지. cv2.inwrite(경로, return값)으로 저장
def cropImage(img, img_gray):
    #b,g,r = cv2.split(img)
    #image = cv2.merge([r,g,b])

    blur = cv2.GaussianBlur(img_gray, ksize=(1,1), sigmaX=0)
    #ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(blur, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #total = 0

    contours_xy = np.array(contours)

    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0])
            x_min = min(value)
            x_max = max(value)
 
    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1])
            y_min = min(value)
            y_max = max(value)

    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min

    img_trim = img[y:y+h, x:x+w]

    # cv2.imwrite(url, img_trim)
    return img_trim
