import numpy as np
from scipy.sparse import linalg as splinalg
from sklearn.cluster import KMeans


def cluNcut(W, c):
    """
    使用归一化图割方法获取最后的聚类结果
    :param W: 一个 n * n 的亲和度矩阵，n 是样本点的数量
    :param c: 聚类个数，子空间个数
    :return: 聚类后的标签
    """

    # 邻接矩阵
    W = np.array(W)
    n = W.shape[0]
    # 将主对角线元素置0：np.diag()提取主对角线元素，返回向量；np.diagflat() 创建对角矩阵
    W = W - np.diagflat(np.diag(W))
    # 重新构建相似度矩阵 W，用于度量数据点之间的相似性或距离。
    W = (W.T + W) / 2
    # 构造度矩阵
    D = np.diag(1/(np.sqrt(np.sum(W, axis=0) + 1e-7)))
    # 归一化的拉普拉斯矩阵
    Lmat = np.eye(n) - D @ W @ D

    # 设置 opts 参数
    opts = {'tol': 1e-8, }
    # 计算特征值和特征向量
    vals, vecs = splinalg.eigs(Lmat, k=c, which='SR', maxiter=30000, tol=1e-8)
    # 将特征向量按列排列
    vecs = np.real(vecs[:, 0:c])
    V = D @ vecs

    # 使用Kmeans进行聚类
    k_means = KMeans(n_clusters=c, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001,  verbose=0, random_state=None, copy_x=True)
    k_means.fit(X=V)
    y_predict = k_means.labels_
    return y_predict





