# ML_AI_Model
OO7D 머신러닝과 인공지능 모델

1. clothes_classification_extended_test.model: 옷 종류 판별 CNN 학습 모델

            * clothes_identification에세 해당 모델의 저장 경로를 입력해 사용한다
            * 모델 적용은 파일 내부에 저장된 것이 아닌 해당 '파일'을 그대로 사용한다

2. clothes_identification: 옷 이미지 종류 판별.

    - clothesIdentify(img) : 입력-Image.open(경로)으로 읽은 이미지, 출력-옷 종류 label
    
            * 학습된 CNN model이 저장된 경로 확인 후 수정
            * 현재 출력은 옷 종류가 label(int)로 출력됨. 한글 출력이나 영어 출력을 원하면 수정 필요

3. color_identification: 옷 이미지 색 판별.

    - colorClustering(img): 입력-cv2.imrerad(경로)로 읽은 이미지, 출력-판별된 색 hex값(str, list) & 색 가중치(int, list)
    
    - colorIdentify(hex_colors, color_percent): hex_colors-색 hex값(str, list), color_percent-색 가중치(int, list), 출력-가중치가 가장 높은 색 2개(int, list) & 가중치%(int, str)
    
            * 현재 색 종류 출력은 label(int, list)로 출력됨. 한글 출력이나 영어 출력을 원하면 수정 필요
            * colorClustering과 같이 쓸 경우 위 함수의 출력을 가공 없이 그대로 사용 가능

4. image_crop: 옷 이미지의 윤곽을 판별해 사진의 여백을 잘라 다시 저장해줌

    - cropImage(img, img_gray): img-cv2.imread(경로)로 읽은 이미지, img_gray-cv2.imread(경로, cv2.IMREAD_GRAYSCALE)로 읽은 흑백이미지, 출력-crop한 이미지. cv2.inwrite(경로)저장 필요
    
            * 옷의 색 판별의 정확도를 높이기 위한 코드이다. 해당 코드를 실행 후 color_identification 실행이 바람직하다
            * front에서 사진 업로드 경고 문구만으로 충분히 사용자가 배경을 배제한 사진을 촬영할 수 있으면 코드를 사용하지 않는다
            * 배경이 복잡하면(벽지 문양, 가구 장식 등) 제대로 작동하지 않는다. 동작 시간과 코드의 낭비가 예상된다면 사용하지 않는다

5. recomendation_TOB: 선택한 상/하의에 어울리는 하/상의의 색 추천

    - recomendationTOB: target-추천 대상 고객의 history DB, most-추천 대상 고객과 상관관계 높은 '타 고객'의 history DB, least-추천 대상 고객과 상관관계 낮은 '타 고객'의 history DB, most_cus_id-추천 대상 고객과 상관관계 높은 '타 고객'의 id list, least_cus_id-추천 대상 고객과 상관관계 낮은 '타 고객'의 id list, most_weight-추천 대상 고객과 상관관계 높은 '타 고객'의 correlation list, least_weight-추천 대상 고객과 상관관계 낮은 '타 고객'의 correlation list, chosen-사용자가 고른 상/하의 색 label, topObottom-사용자가 고른 옷이 상의이면 0, 하의이면 1, 출력-추천할 상의 혹은 하의의 색 label

            * recomendation_calculate_similarity 함수를 수행 후 해당 함수를 수행한다
            * most의 경우 상관관계가 높은 '타 고객'의 history DB를 하나로 합친 후 넘긴다
            * 상관관계가 음수인 고객이 없을 경우 least는 빈 변수를, least_cus_id, least_weight는 빈 list를 넘긴다
            * topObottom이 0이면 고객은 상의를 선택해 하의를 추천받기를 원하는 상태이며, 1이면 고객은 하의를 선택해 상의를 추천받기 원하는 상태이다

6. recomendation_calculate_similarity: 추천 대상 고객과 타 고객 관의 상관계수 계산

    - recCusSimilarity(cus, target_id, target_color, topObottom): cus-추천 대상 고객과 '타 고객'의 착장history DB를 concate한 DataFrame, target_id-추천 대상 고객의 id, target_color-고객이 선택한 상/하의의 색 label, topObottom-고객이 선택한 옷이 상의이면0, 하의이면1 입력, 출력-상관계수가 높은 고객 5명의 id, 상관계수 높은 고객의 상관계수, 상관계수 낮은 고객 1명의 id, 상관계수 낮은 고객의 상관계수
 
            * cus는 recomendation_customer_select에서 반환받은 '타 고객'의 history DB와 추천 대상 고객의 history DB를 합친 DataFrame을 받는다
            * topObottom이 0이면 고객은 상의를 선택해 하의를 추천받기를 원하는 상태이며, 1이면 고객은 하의를 선택해 상의를 추천받기 원하는 상태이다
            * 출력은 모둔 int의 list이다
            * 상관계수가 낮은 고객의 경우 상관계수가 음수인 고객이 있는 경우에만 return한다. 상관계수가 음수인 고객이 없으면 빈 str을 return한다

7. recomendation_clothes_select: 옷장에 있는 옷 중 추천받은 색 기준 옷 한 벌 추천

    - recClothesSelect: cloeset-사용자의 옷장DB Df, recColor-추천받은 색깔 label int, style-추천할 옷 스타일, topObottom-고객이 선택한 옷이 상의이면0, 하의이면1 입력, 출력-추천하는 상/하의의 옷의 id list

            * style은 상/하의 추천인 경우 사용자가 선택한 상/하의의 스타일 label을, 랜덤추천인 경우 사용자가 선호하는 스타일 label을 입력한다
            * 스타일은 정장(0), 세미정장(1), 캐주얼(2), 스포츠(3) 이다.
            * 만약 해당 스타일의 해당 색상이 없는 경우 각 스타일 label의 -1~+1 label을 가지는 스타일까지 모두 검사한다
            * 랜덤추천의 경우 해당 함수를 상의와 하의 총 두 번 실행한다
            * topObottom이 0이면 고객은 상의를 선택해 하의를 추천받기를 원하는 상태이며, 1이면 고객은 하의를 선택해 상의를 추천받기 원하는 상태이다
            * return 받은 옷의 id 개수는 1개 이상일 수 있다. 해당 경우에는 이 중 랜덤으로 한 벌을 사용자에게 추천한다
            * return 받은 옷의 id 개수가 1개 이하, 즉 0개일 경우 인터넷 쇼핑몰 크롤링 자료를 표시한다

8. recomendation_customer_select: 상의와 하의 추천시 추천받을 '타 고객' 선정

    - recCusSelect(cus, targetID): cus-전체 고객에 대한 정보가 들어간 DB, targetID-추천 대상 고객의 ID, 출력-추천 받은 '타 고객'의 id

            * cus에는 고객id, 성별, 나이, 선호스타일 데이터가 포함되어 있어야 한다
            * filtering순서는 성별→나이→선호스타일 순서이다. 선호스타일의 경우 나이까지 filtering해 남은 '타 고객'이 10명 이하일 경우 filtering하지 않는다
            * 출력으로 받은 '타 고객'id를 바탕으로 이들의 history DB와 추천 대상 고객의 history DB를 합쳐 recomendation_calculate_similarity의 입력으로 사용한다

9. recomendation_random: 상의와 하의의 색을 동시에 추천

    - recomendationRandom(combination): 입력-고객의 착장history DB, 출력-추천 상의 색 label(int)과 하의 색 label(int)

            * 추천에 사용하는 column: 상의 색, 하의 색
            * history DB 정보를 받아올 때 DataFrame으로 받을 것
            * 현재 추천 색 출력은 label(int)로 출려됨. 한글 출력이나 영어 출력을 원한다면 수정 필요


10. weather_classifier.joblib: 날씨 추천 Random Forest 학습 데이터

11. weather_identification: 현재 날씨에 따라 입을 수 있는 옷과 입을 수 없는 옷 필터링

    - todayTemp(cityNum): 입력-현재 날씨를 받을 지역, 출력-해당 지역의 현재 날씨

            * 추천받을 수 있는 지역: 서울(0), 부산(1), 충북(2), 충남(3), 대구(4), 대전(5), 강원(6), 광주(7), 경기(8), 경북(9), 경남(10), 인천(11), 제주(12), 전북(13), 전남(14), 세종(15), 울산(16)
            * 회원가입 시 고객의 거주 지역을 입력받고, 고객정보 수정에서 지역을 수정할 수 있다
            * 지역 구분을 무시할 경우 cityNum은 0으로 고정, 서울 기온을 고정을 받는다
            * 사용한 날씨 api는 openweathermap이다. 코드 사용시 적절한 apikey를 작성한다

    - weatherIdentify(closet, temp): target-추천받는 고객의 옷장 DB, temp-현재 날씨, 출력-입을 수 있는 옷의 id(list, int)

            * 학습된 RandomForest model이 저장된 경로 확인 후 수정
            * todayTemp와 같이 쓰이는 경우 위 함수의 출력을 가공 없이 temp에 그대로 사용 가능
