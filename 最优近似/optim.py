import pandas as pd
import torch
import function_set as fc
import json

def optim(Cx, X, Y):#X是粗糙集,Y分化啦
    dict = []
    te = []
    for i, x in enumerate(Cx):

        value_x = len(fc.num_n(X, x)) / (len(x) - len(fc.num_n(X, x)))
        dict.append([value_x, x])

    l = sorted(dict, key=lambda tem: tem[0])
    C = []

    while len(l) != 0:
        max_x = l[-1]#最
        down = fc.xiajinsi(X, Y)
        value_add = add(len([j for ite in down for j in ite]), len(X), len(fc.num_n(X, max_x[1])),\
                        (len(max_x[1]) - len(fc.num_n(X, max_x[1]))))
        # print(value_add, len([j for ite in down for j in ite])/len(X)\
        #                 ,len([j for ite in down for j in ite]), len(X), len(fc.num_n(X, max_x[1])),\
        #                 (len(max_x[1]) - len(fc.num_n(X, max_x[1]))))
        idx = []
        for i in range(len(l)):
            if l[i][0] < value_add:
                idx.append(i)
        # print(idx, l)
        if len(idx) != 0:
            l1, l2 = idx[0], idx[-1] + 1
            del l[l1: l2]
        if len(l) != 0:
            del l[-1]
        C.append(max_x)
    C = [tem[1] for tem in C]
    return C


def add(X1, Y1, X2, Y2):
    return (X1+X2) / (Y1+Y2)






if __name__ == '__main__':
    dict = {1:2, 3:4, 5:2}

    x = [1, 2, 3]
    del x[0: 2]
    print(x)
    x = json.dumps(x)
    dict[x] = 4
    i, j, opti_sim, orignal_sim = 1, 2, 3, 4
    d = [[123], [2, 5, 6]]
    with open('./tes.json', 'w', encoding='utf8') as f:
        x = [i, j, opti_sim, orignal_sim]
        json.dump(('result=', x, 'xiajisnsi=', d), f)
        f.close()
    data_set = pd.DataFrame([[1, 2, 3]])
    print(data_set.iloc[:, :2])

