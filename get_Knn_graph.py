import numpy as np
from scipy.spatial.distance import pdist, squareform
# from scipy.sparse import csr_matrix, tril


def getKnnGraph(X, k):
    """
    构建KNN相似度图
    :param X: 数据矩阵，n列样本，d行特征
    :param k: 亲和度图的邻居数量
    :return:
    """

    # 获取数据的列数，即样本数
    X = np.array(X)
    num_simple = X.shape[1]

    # 依照原matlab代码中的函数实现，主要功能是计算各点之间的欧式距离。大小为样本数量乘以样本数量
    dist_Min = squareform(pdist(X.T))

    # 对矩阵升序排列，按列方向进行排序，返回排序后的距离矩阵和位置。大小为样本数量乘以样本数量
    dist_Min = np.array(dist_Min)
    dist_index = np.argsort(dist_Min, axis=0)

    # 初始化邻居矩阵
    neigh_Mat = np.zeros((num_simple, num_simple))
    # 构建 0-1 权重的 KNN 相似度图（对称矩阵）：逐行处理，大小为样本数量
    for i in range(num_simple):
        # print(dist_index[0:k, i])
        # dist_index[0:k, i]：选择距离矩阵中的前k行，构成了 k * n 的矩阵，再逐列进行选择。
        neigh_Mat[dist_index[0:k, i], i] = 1
        # 邻居矩阵是对称矩阵，上述是对列进行处理，下述是对行进行处理。
        neigh_Mat[i, dist_index[0:k, i]] = 1

    # 使用 scripy 库中的函数进行处理
    # neigh_Mat = csr_matrix(neigh_Mat)
    # neigh_Mat = tril(neigh_Mat)
    # 使用 numpy 库中的函数进行处理，选择矩阵对角线以下元素
    neigh_Mat = np.tril(neigh_Mat, -1)

    # 实现 matlab中的 ind2sub() 函数
    # 查找邻居矩阵中大于0的位置信息，也就是与样本 i 最近的 k 个邻居的位置信息
    nz_index = np.where(neigh_Mat > 0)
    # 返回行的索引和列的索引
    nzr, nzc = nz_index[0], nz_index[1]
    # 按垂直方向（行顺序）堆叠数组构成一个新的数组
    pairs = np.vstack((nzr, nzc))
    # 将非0元素的位置矩阵按照原来的列号升序排序
    pairs = pairs.T[np.argsort(pairs.T[:, 1])].T
    # 对原矩阵一共 n 行按照行号进行排序
    for i in range(num_simple):
        # 获得相同列号下的坐标
        temp = pairs.T[pairs.T[:,1] == i]
        # 对相同列号下的坐标按照行号升序排列
        temp_sort = temp[np.argsort(temp[:, 0])]
        # 用排好序的索引替换原索引
        pairs.T[pairs.T[:, 1] == i] = temp_sort

    # 统计成对的邻居数量
    numpairs = len(nz_index[0])
    # 与非零元素个数相等的行向量，元素全为1。
    # 可以在使用时直接赋值为1，没有必要创建1向量
    wcost = np.ones(numpairs)

    return pairs, wcost, numpairs


"""
1、pdist()是一个计算距离的函数，默认是欧氏距离，计算后的结果是一个上三角形矩阵，其中对角线为0。squareform()函数是将pdist()函数返回的结果组织称对称矩阵。
从第一行开始取值，返回一个数组，变成一个稀疏矩阵，同时spuareform()函数还可以进行逆运算，把一个稀疏矩阵生成一个非稀疏矩阵。
matlab中的方法和scipy中的方法用法相同。
2、scipy.sparse.tril(A, k=0, format=None)
以稀疏格式返回矩阵的下三角部分
返回矩阵 A 的 k-th 对角线之上或之下的元素。
k = 0 对应于主对角线
k > 0 在主对角线上方
k < 0 在主对角线下方
3、​matlab中的 ind2sub 函数
[row,col] = ind2sub(sz,ind)：返回数组的 row 和 col；先按照col从小到大排列，然后按照row从小到大排列。
经常搭配[ind] = find(Mat>0)使用，ind是大小为 sz 的矩阵的线性索引；matlab的find函数的策略是从第1列由上而下，接着第2列由上而下…的索引，
而且是打平了的索引，不是坐标，返回的是按照序号升序的线性索引。
此处，sz 是包含两个元素的向量，其中 sz(1) 指定行数，sz(2) 指定列数

"""
