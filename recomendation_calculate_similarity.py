import pandas as pd

# 타겟 고객과 비슷한 상관도를 가진 5명과 상관도가 가장 낮은 1명 추출
# cus: recCusSelect에서 구한 고객과 타겟의 착장history DB DataFrame (모든 DB concate)
# target_id: 추천할 타겟 고객의 고객ID
# target_color: 추천받을 상의 혹은 하의의 색 label
# topObottom: target_color가 상의면 0, 하의면 1 입력
# 출력: 상관도가 가장 높은 5명의 고객 ID list와 상관도 list, 상관도가 가장 낮은 1명의 고객 ID list와 상관도 list
def recCusSimilarity(cus, target_id, target_color, topObottom):
    # cus = pd.read_csv(target_cus)  # 비슷한 선호의 고객과 타겟 고객의 착장history (합침)

    if topObottom == 0:  # 상의 색 기준 하의 추천
        cus_color = cus[cus['top_cls'] == target_color]
        cus_color.drop(['top_cls'], axis=1, inplace=True)
    else:  # 하의 색 기준 상의 추천
        cus_color = cus[cus['bottom_cls'] == target_color]
        cus_color.drop(['bottom_cls'], axis=1, inplace=True)

    # history에서 타켓의 데이터 추출
    target_cus = cus_color[cus_color['id'] == target_id]
    cus_color.drop(cus_color[cus_color['id'] == target_id].index, inplace=True)

    #target_cus의 착장 수가 10개 이상인 경우만 최신 데이터 slicing.
    longest = 0
    for i in cus_color['id'].unique():
        if len(cus_color[cus_color['id'] == i]) > longest:
            longest = len(cus_color[cus_color['id'] == i])

    target_cus[len(target_cus) - int(len(target_cus) - (longest/2)):]

    if len(target_cus) > 10:
        target_cus[len(target_cus) - int(len(target_cus) - (longest/2)):]
    else:
        pass

    # slicing한 target_cus를 기존 cus_color와 합침
    cus_top2 = pd.concat([cus_color, target_cus])
    cus_top2.reset_index(drop=True, inplace=True)

    # similarity계산(피어슨 상관계수)
    cus_top2 = cus_top2.groupby(['id', 'bottom_cls'], as_index=False).mean()
    cus_pivot = pd.pivot(cus_top2, index='bottom_cls', columns='id', values='rating')
    cus_sim = cus_pivot.corr()

    # 상관도가 비슷한 5명의 고객ID 추출
    target_corr = pd.DataFrame(cus_sim[target_id]).dropna()
    target_corr.drop(index=target_id, inplace=True)
    corr_plus = target_corr[target_corr[target_id] > 0.3]  # 상관계수가 0.3이상인 고객만 추출

    if min(target_corr[target_id]) < 0:  #상관관계가 음수를 보이는 사용자가 있을 경우, 1명 추출
        corr_minus = target_corr[target_corr[target_id] == min(target_corr[target_id])]
        minus_list = corr_minus[target_id].tolist()
    else:
        minus_list = []

    #상관계수가 높은 5명 추출. 양의 상관계수가 5명 이하이면 생략
    plus_list = corr_plus[target_id].tolist()
    if len(corr_plus) > 5:
        big = [max(plus_list)]  #가장 높은 상관계수
        plus_list.remove(max(plus_list))
        high = 0
        # 가장높은 상관계수 5개 추출
        for count in range(4):
            for i in plus_list:
                if i > high:
                    high = i
            big.append(high)
            plus_list.remove(high)
            high = 0
    else:
        big = plus_list

    #가장 높은 상관계수를 가지는 5명의 고객 id를 return
    plus_cus_id = []
    for i in range(len(big)):
        plus_cus_id.append((corr_plus[corr_plus[target_id] == big[i]]).index[0])
    
    if len(corr_minus) > 0:
        minus_cus_id = [corr_minus.index[0]]
    else:
        minus_cus_id = []

    return plus_cus_id, big, minus_cus_id, minus_list
