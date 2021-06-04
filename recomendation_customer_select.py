import pandas as pd
from datetime import datetime

# target 고객과 성별, 나이, 선호에 따라 추천받을 고객 filtering(5+1추천시)
# cus: 전체 고객 DB DataFrame
# targetID: 추천해 줄 고객의 고객ID
# 출력: 성별, 나이, 선호가 동일한 다른 고객의 ID list
def recCusSelect(cus, targetID):
    # cus = pd.read_csv(customer)  # 전체 고객정보 DB 경로

    # 타겟 고객데이터 
    target = cus[cus['id'] == targetID]
    target_index = cus[cus['id'] == targetID].index[0]
    cus.drop(index=target_index, inplace=True)
    target_dict = target.to_dict()

    # 타겟과 동일 성별 filter
    target_dict['gender'][target_index]
    target_cus = cus[cus['gender'] == target_dict['gender'][target_index]]

    # 타겟과 동일 나이대 filter
    age = datetime.today().year - target_cus['DoB']
    target_cus['age'] = age
    target_age = datetime.today().year - target_dict['DoB'][target_index]
    target_age = int(target_age / 10) * 10
    target_age_10 = target_age + 10
    target_cus = target_cus[(target_cus['age'] >= target_age) & (target_cus['age'] < target_age_10)]

    # 타겟과 동일 선호스타일 filer
    # 필터링 할 고객 수가 10명 미만이면 선호스타일 필터링 X
    if (len(target_cus) < 10):
        pass
    else:
        target_cus = target_cus[target_cus['preference'] == target_dict['preference'][target_index]]

    return target_cus['id'].tolist()