import numpy as np
import json
import function_set as fc
import torch
import pandas as pd
import optim
from multiprocessing import Pool
data_set = pd.read_csv('../data_train/car.csv')
# ''''''
replace_set = {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4, '5more': 5, 'more': 5, 'small': 1, 'big': 2,
               'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}
data_set.columns = list(range(7))
data_set = data_set.replace(replace_set)
data_set = data_set.astype('float')
L = np.arange(0, data_set.shape[1] - 1, dtype=float)

set_all_ar = fc.lidu(L, 3, data_set.shape[1] - 1) # 求出属性组和
'''多粒度搜索'''
X = data_set.iloc[:, data_set.shape[1] - 1]
X = fc.pandas_to_torch(X)
X = fc.dengjialei(X)
X = [x for x in X]
# side线程
def test(lis):
    dataset_den = np.array(data_set.iloc[:, lis])
    data = torch.tensor(dataset_den) # 原数据
    Y = fc.dengjialei(data) # 元数据的等价类
    dataset = [y for y in Y] #
    for j in range(len(X)):
        d = fc.xiajinsi(X[j], dataset)  # 下近似3
        Dx, Cx = fc.DxCx(X[j], dataset)
        C = optim.optim(Cx, X[j], dataset)

        if len(C) != len(Cx):
            opti_sim = [i1 for ite in C + d for i1 in ite]
            orignal_sim = [j for jte in Cx + d for j in jte]
            opti_sim = fc.similarity(opti_sim, X[j])
            orignal_sim = fc.similarity(orignal_sim, X[j])
            # print('C=', C, '\n', '\n', 'Cx=', Cx, '\n', 'lenC,lenCx', len(C), len(Cx))
            # print('reusl:', lis, opti_sim, orignal_sim)
            return  ['result',lis ,opti_sim, orignal_sim, 'xiajinsi', d,
                     'lenC,lenCx', len(C), len(Cx)]

if __name__ == '__main__':
    pool = Pool(6)
    final_result = pool.map(test, set_all_ar)
    pool.close()
    with open('../new_result_Data/car_result.json', 'w', encoding='utf8') as f:
        json.dump(final_result, f)
        f.close()





