import pandas as pd
import numpy as np
import joblib
import requests
import json

# 현재 기온을 날씨api로부터 받아옴
# cityNum: 온도를 받은 지역에 할당된 숫자
# return: 현재 기온 (int)
def todayTemp(cityNum):
    apikey = "65457990dc620f30dad21bc521e49123"  # api접근 key

    cities = ["Seoul,KR", "Busan, KR", "Chungcheongbuk-do, KR", "Chungcheongnam-do, KR", "Daegu, KR", "Daejeon, KR",
          "Gangwon-do, KR", "Gwangju, KR", "Gyeonggi-do, KR", "Gyeongsangbuk-do, KR", "Gyeongsangnam-do, KR", "Incheon, KR",
          "Jeju, KR", "Jeollabuk-do, KR", "Jeollanam-do, KR", "Sejong, KR", "Ulsan, KR"]

    api = "http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}"

    # Kelvin to Celcius
    k2c = lambda k: k - 273.15

    url = api.format(city=cities[cityNum], key=apikey)
    r = requests.get(url)
    data = json.loads(r.text)
    temp = round(k2c(data["main"]["temp"]))

    return temp

# 옷장에 저장된 옷들 중 오늘 기온에 입을 수 있는 옷 filter
# target: 사용자의 옷장 데이터 경로
# temp: 날씨 api에서 가져온 오늘 기온
# return: 오늘 기온에 입을 수 있는 옷들의 id list
def weatherIdentify(target, temp):
    model = joblib.load('weather_classifier.joblib')  # RandomForest 학습 데이터 저장 경로
    closet = pd.read_csv(target)

    id_list = closet['id'].tolist()
    type_list = closet['type'].tolist()
    weather_list = closet['weather'].tolist()

    result_id = []
    for i in range(len(closet)):
        buf = []
        buf.append(temp)
        buf.append(type_list[i])
        buf.append(weather_list[i])
        
        buf = np.array([buf])
        res = model.predict(buf)[0]
        
        if res == 9:
            result_id.append(id_list[i])

    return result_id