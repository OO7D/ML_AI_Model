# ML_AI_Model
OO7D 머신러닝과 인공지능 모델

1. clothes_identification: 옷 이미지 종류 판별.

    - clothesIdentify(url) : 입력-이미지 저장 경로, 출력-옷 종류
    
            * 학습된 CNN model이 저장된 경로 확인 후 수정
            * 현재 출력은 옷 종류가 영어(str)로 출력됨. 한글 출력이나 label 숫자로 출력을 원하면 수정 필요

2. color_identification: 옷 이미지 색 판별.

    - colorClustering(url): 입력-이미지 저장 경로, 출력-판별된 색 hex값(str, list) & 색 가중치(int, list)
    
    - colorIdentify(hex_colors, color_percent): hex_colors-색 hex값(str, list), color_percent-색 가중치(int, list), 출력-가중치가 가장 높은 색 2개(str, list) & 가중치%(int, str)
    
            * 현재 색 종류 출력은 영어(str)로 출력됨. 한글 출력이나 label 숫자로 출력을 원하면 수정 필요
            * colorClustering과 같이 쓸 경우 위 함수의 출력을 가공 없이 그대로 사용 가능

3. image_crop: 옷 이미지의 윤곽을 판별해 사진의 여백을 잘라 다시 저장해줌

    - cropImage(url): 입력-이미지 저장 경로, 출력-X
    
            * 옷의 색 판별의 정확도를 높이기 위한 코드이다. 해당 코드를 실행 후 color_identification 실행이 바람직하다
            * front에서 사진 업로드 경고 문구만으로 충분히 사용자가 배경을 배제한 사진을 촬영할 수 있으면 코드를 사용하지 않는다
            * 배경이 복잡하면(벽지 문양, 가구 장식 등) 제대로 작동하지 않는다. 동작 시간과 코드의 낭비가 예상된다면 사용하지 않는다

4. recomendation_random: 상의와 하의의 색을 동시에 추천

    - recomendationRandom(target): 입력-고객의 착장history DB, 출력-추천 상의 색(str)과 하의(str)

            * 추천에 사용하는 column: 상의 색, 하의 색
            * history DB 정보를 받아올 때 pandas로 받을 것
            * 불러올 파일의 확장자가 csv가 아닌 경우 알맞은 함수로 변경한다
            * 현재 추천 색 출력은 영어(str)로 출려됨. 한글 출력이나 label 숫자로 출력을 원한다면 수정 필요
