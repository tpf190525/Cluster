import numpy as np
from munkres import Munkres
import time
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from skfeature.smr.smr import smr
from skfeature.smr.clu_ncut import cluNcut


def best_map(label1, label2):
    """
    对聚类后的到预测标签重排
    :param label1: 原始数据标签
    :param label2: 经过聚类后得到的预测标签
    :return: 经过重排后的预测标签
    """

    # 对于预测标签中的每个元素加1，因为预测标签是从0开始计算的，真实标签是从1开始算的
    # label2 = label2 + 1
    # 对于一维数组或者列表去重并按元素由大到小返回一个新的无元素重复的元组或者列表
    L1 = np.unique(label1)
    L2 = np.unique(label2)
    # 统计唯一化后的标签。基本就是该数据集原本的类别数量
    uclass1 = len(L1)
    uclass2 = len(L2)
    uclass = np.maximum(uclass1, uclass2)
    # 创建标签组合矩阵
    L = np.zeros((uclass, uclass))
    # 计算标签数量
    for i in range(uclass1):
        # 统计唯一化后标签在原标签序列中的位置并赋值给位置向量。相当于三元运算符的作用
        index_class1 = label1 == L1[i]
        # 将布尔类型转换为float类型
        index_class1 = index_class1.astype(float)
        for j in range(uclass2):
            index_class2 = label2 == L2[j]
            index_class2 = index_class2.astype(float)
            # 计算真实标签与预测标签的数据重合数量。重合数量越高则预测标签属于真实标签的概率越大
            L[i, j] = np.sum(index_class1 * index_class2)

    # 实例化Kuhn-Munkres算法
    m = Munkres()
    # 矩阵L的定义是行为原标签，列为预测标签。转置之后表示预测标签与真实标签的对应关系
    index = m.compute(-L.T)
    # index 是计算后的类别映射表。
    index = np.array(index)
    # 类别映射表中的类别是从0开始计算的
    # index = index + 1
    newLabel = np.zeros(label2.shape, dtype=int)
    for i in range(uclass2):
        for j in range(len(label2)):
            if label2[j] == index[i, 0]:
                newLabel[j] = index[i, 1]

    return newLabel+1


def run():

    # load data。USPSdata_20_uni
    mat = scipy.io.loadmat('../data/jaffe.mat')
    X = mat['fea']    # data
    X = X.astype(float)
    y = mat['gnd']    # label
    y = y[:, 0]

    start_time = time.time()

    # 运行主函数，获得亲和度矩阵 W
    W = smr(X.T, a=2**-16, k=10, aff_type="J2", r=0.01)
    n = W.shape[1]
    W2 = W
    for i in range(n):
        W2[:, i] = W[:, i] / (np.max(np.abs(W[:, i])) + 1e-7)

    label = cluNcut(W2, 10)

    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = 0
    acc_total = 0
    for i in range(0, 20):
        # calculate NMI
        nmi = normalized_mutual_info_score(y, label)
        # calculate ACC
        y_permuted_predict = best_map(y, label)
        acc = accuracy_score(y, y_permuted_predict)

        nmi_total += nmi
        acc_total += acc

    e1 = float(acc_total) / 20
    e2 = float(nmi_total) / 20

    end_time = time.time()
    all_time = end_time - start_time
    e3 = all_time

    # 返回三种评估指标
    return e1, e2, e3


e1, e2, e3 = run()
print("e1, e2, e3", e1, e2, e3)