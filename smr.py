import numpy as np
from scipy.linalg import solve_sylvester

from skfeature.smr.get_Knn_graph import getKnnGraph


def smr(X, a, r, k, aff_type):
    """
    算法主函数
    :param X: 数据集，d行特征，n列样本
    :param a: 表示误差前面的参数
    :param r: 构造亲和度矩阵时的参数
    :param k: 构建亲和度图时的邻居数量
    :param aff_type: 亲和度矩阵的计算方式
    :return: 亲和度矩阵L
    """

    # 获取数据集的特征数和样本数
    num_simple = X.shape[1]

    # 创建 KNN 相似度矩阵
    pairs, wcost, numpairs = getKnnGraph(X, k)
    # 根据KNN图创建相应的拉普拉斯矩阵
    # 这样构造拉普拉斯矩阵的依据是什么？
    R = np.zeros((num_simple, numpairs))
    # 对于每个配对，将 wcost[i] 的值放在相应的点对的位置，一个点的权重为正，另一个点的权重为负。
    for i in range(numpairs):
        # pairs[0, i] 表示第 i 个配对中的第一个点的索引。
        R[pairs[0, i], i] = wcost[i]
        # pairs[1, i] 表示第 i 个配对中的第二个点的索引。
        R[pairs[1, i], i] = -wcost[i]
    # 进行归一化，将权重缩放到一定的范围内。
    R = R / (k - 1)
    # 构造拉普拉斯矩阵，拉普拉斯矩阵可以通过相邻节点的连接关系来度量节点之间的关联性。乘以 0.5 是为了保持对称性。
    RRT = 0.5 * R @ R.T

    XTX = X.T @ X
    elpson = 0.001
    A = a * XTX
    B = RRT + elpson * np.eye(num_simple)
    # 为什么matlab代码中此处要设置成负的？
    C = a * XTX

    # 解 Sylvester 方程。若C为负，则解的正负号与matlab中的相反。
    J = solve_sylvester(A, B, C)

    # 根据指定的方式构建亲和度矩阵
    if aff_type == 'J1':
        L = (np.abs(J) + np.abs(J.T)) / 2
    elif aff_type == 'J2':
        # 对矩阵按列求2范数，将矩阵每列中的元素进行平方求和然后开根号
        nX = np.sqrt(np.sum(X ** 2, axis=0))
        # 将 nX 从num_simple维的列向量进行转换
        nX = np.reshape(nX, (1,num_simple))
        L = np.abs((J.T @ J) / (nX.T @ nX)) ** r
    elif aff_type == 'J3':
        L = np.abs(J.T @ J) ** r

    return L
