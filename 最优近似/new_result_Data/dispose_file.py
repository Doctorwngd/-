import numpy as np
import pandas as pd
import json

with open('./mush_result.json', 'r', encoding='utf8') as f:
    data_list = json.load(f)
    f.close()
data_set = []
for t1 in data_list:
    if t1 is None:
        continue
    else:
        attribute, opti_s, previ_sim, len_of_d, C, Cx = t1[1], t1[2], t1[3], len(t1[5]), t1[7], t1[8]
        data_enhance = opti_s / previ_sim
        data_enhance = str(round(data_enhance, 2) * 100)
        data_set.append([opti_s, previ_sim, data_enhance, len_of_d, attribute, C, Cx])

data_set = sorted(data_set, key=lambda X: eval(X[2]), reverse=True)
data_result = [data_set[0]]
for row in data_set:
    if row[2] != data_result[-1][2]:
        data_result.append(row)
#
#
result = pd.DataFrame(data_result, columns=['optim_', 'prev', 'enhance_rate:%', 'len_d',
                                            'attribute', 'len_Cx', 'len_C'])
result.to_csv('./data_show/m_data.csv')
# result.to_csv('./rate/car_rate.csv')
# result.to_csv('./rate/mushroom_rate.csv')