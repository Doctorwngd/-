import numpy as np
import json
import function_set as fc
import torch
import pandas as pd
import optim
from multiprocessing import Pool

pd.set_option('display.max_columns', None)
data_set = pd.read_csv('../data_train/bank.csv', sep=';')
data_set = data_set.select_dtypes(include='object') # 选择object
# ''''''
def find_dict(data_set): #寻找replace_set
    '''find out every classes of columns,replaced by number'''
    data_set = data_set.T.values.tolist()
    dic = []
    for column in data_set:
        atribute = dict([[tem, i+1] for i, tem in enumerate(set(column))])
        dic.append(atribute)

    empty_dict = {}
    for ite in dic:
        empty_dict.update(ite)
    return empty_dict


replace_set = find_dict(data_set)
data_set = data_set.replace(replace_set)
data_set = data_set.astype('float')

L = np.arange(0, data_set.shape[1] - 1, dtype=float)
set_all_ar = fc.lidu(L, 4, 7) # 求出属性组和
'''多粒度搜索'''
X = data_set.iloc[:, data_set.shape[1]-1]
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
    with open('../new_result_Data/bank_result.json', 'w', encoding='utf8') as f:
        json.dump(final_result, f)
        f.close()






