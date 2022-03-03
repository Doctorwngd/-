import numpy as np
import torch
import pandas as pd
import json
import itertools

def dengjialei(X):#求等价类，表上类别号
    y = torch.zeros(X.shape[0])
    y = y.unsqueeze(1)
    # X = torch.cat((X, y), 1)
    k = 0#种类
    for i in range(len(X)):#得到相同类标记的数组
        if y[i] != 0:
            continue
        k += 1
        for j in range(i, len(X)):
            if X[i].equal(X[j]):
                y[j] = k

    # X = torch.cat((X, y), 1)
    set_y = set([i.item() for i in y])
    for tem in set_y:#
        result = [i for i, x in enumerate(y) if x == tem]
        # i是对象编号，x是类别编号
        yield result # 返回一个嵌套列表，元素是对象的编号

def num_n(X, Y):#求X和Y的交
    if torch.is_tensor(X):
        X = [i.item() for i in X]
    if torch.is_tensor(Y):
        Y = [i.item() for i in Y]
    set_n = [val for val in X if val in Y]
    return set_n

def num_u(X, Y):#求X和Y的并
    if torch.is_tensor(X):
        X = [i.item() for i in X]
    if torch.is_tensor(Y):
        Y = [i.item() for i in Y]
    set_u = list(set(X + Y))
    return set_u

def similarity(X, Y):#求X和Y的相似度
    simlar = len(num_n(X, Y)) / len(num_u(X,Y))
    return simlar

def xiajinsi(X, Y):#下近似
    down = []
    for item in Y:
        calculate = [0 if i in X else 1
                     for i in item]
        cal = sum(calculate)
        if cal == 0:
            down.append(item)
        else:
            continue
    return down # 返回一个嵌套列表，元素是对象的编号

def baohandu(X, Y):#D(X/Y)
    return len(num_n(X, Y)) / len(Y)

def Alpham(X, Y):#X下近似的实除以X个数
    down = xiajinsi(X, Y)
    alpha = len([j for ite in down for j in ite]) / len(X)
    # print('alpha', len([j for ite in down for j in ite]) / len(X))
    return len([j for ite in down for j in ite]) / len(X)


def DxCx(X, Y):#论文中的两个,X粗糙，Y是分化啦
    Cx = []
    Dx = []
    for y in Y:
        DXY = baohandu(X, y)#包含度
        if DXY > 0 and DXY < 1:
            Dx.append(y)
    for y1 in Dx:
        numn = len(num_n(X, y1))
        if numn / (len(y1) - numn) >= Alpham(X, Y):
            Cx.append(y1)

    return Dx, Cx # 返回一个嵌套列表，元素是对象的编号，次级元素是等价类

def reduction(X):# 传入一个辨识矩阵
    reduction_set = []
    while len(X) != 0:
        X, del_set = sorted(X, key=lambda x:len(x)), []
        x1 = set(X[0])
        for i, L in enumerate(X):
            D_attribute = set(L)
            if len(x1 & D_attribute) == len(x1):# 如果是L的子集的话
                del_set.append(i)
        for i in del_set:# 删除列表中的元素
            X[i] = 'a'
        X = [i for i in X if i != 'a']
        reduction_set.append(x1)
    return reduction_set # 返回一个嵌套矩阵，元素是约简后的属性(还没有求合取)

def bianshijuzhen(X1, X2, Y):#X1，X2传入2等价类，利用标号找出每一个等价类的数据, Y表示原数据
# 直接从等价类中抽出原数据
# X是一个嵌套列表，Y是torch.tensor 类型的数据

    bianshi_x = ((Y[X1[0]] - Y[X2[0]]) != 0).float().numpy().tolist()
    return [i for i, ite in enumerate(bianshi_x) if ite == 1.] # 两个等价类辨识矩阵

def lidu(L, a, b): # 输入一个属性列表L，a和b-1表示粒度长度的最小个数和最大个数
    set_all_attribute = []
    for i in range(a, b): # 得到一部分粒度， 特征个数为3的所有粒度到特征个数到7的所有粒度
        attribute_set = itertools.combinations(L, i)
        set_all_attribute.append(attribute_set)
    set_all_attribute = [list(sigle) for ite in set_all_attribute for sigle in ite]
    return set_all_attribute

def pandas_to_torch(X):
    X = np.array(X)
    X = X.astype(float)
    return torch.tensor(X)


if __name__ == '__main__':
    Y = torch.tensor([[1, 1, 2],
                     [1, 1, 2],
                     [2, 2, 3],
                     [2, 2, 3],
                     [1, 0, 3],
                      [1, 1, 2],
                      [2, 2, 3],
                      [0, 0, 9],
                     ])
    X = dengjialei(Y)
    X = [ite for ite in X]
    print(X)
    x = bianshijuzhen(X[2], X[1], Y)
    y = reduction(x)
    print(x,y)
    # print([i for i in x])
    # X = torch.tensor([1, 2, 3])
    # Y = [2, 3, 4]
    # simlar = similarity(X, Y)
    # print(simlar)
    # X = [1, 2, 3, 4, 5, 6]
    # Y = [[4, 5],[3, 4, 7],[1, 10, 99],[7, 9, 1]]
    # Dx,Cx = DxCx(X, Y)
    # print(Cx)
    # Y = [i for i in x]
    # data = pd.DataFrame(Y)
    # print(data)
    # with open('./data_set.json', 'w', encoding='utf8') as f:
    #     json.dump(Y, f)
    #     f.close()
    # with open('./data_set.json', 'r', encoding='utf8') as f:
    #     l = json.load(f)
    #     print(l)
    #     f.close()
    X = [[1, 2, 4], [1, 2, 3], [3, 4], [3]]
    Y = [[2, 3]]
    if X != Y:
        print('1')
    set1 = []
    for L in X:
        set1.append(set(L))
    print(set1[1] - set1[-1])
    # del_set = [1, 2]
    # for i in del_set:
    #     X[i] = 'a'
    # X = [i for i in X if i != 'a']
    print(reduction(X))