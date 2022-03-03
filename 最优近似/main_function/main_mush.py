import numpy as np
import json
import function_set as fc
import torch
import pandas as pd
import optim
from multiprocessing import Pool
replace_set = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, \
               'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't':20,\
               'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '?': 0}
dataset = pd.read_csv('../data_train/mushroom.csv')
data_set = dataset.replace(replace_set)
L = np.arange(0, data_set.shape[1] - 1, dtype=float)

set_all_ar = fc.lidu(L, 3, 4) # 求出属性组和
'''多粒度搜索'''
X = data_set.iloc[:, -1]
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
    with open('../new_result_Data/mush_result.json', 'w', encoding='utf8') as f:
        json.dump(final_result, f)
        f.close()
