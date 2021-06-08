import torch


# 求x和y的欧氏距离
def euclidean_dist(x, y):
    # x: N x D [300 x 64] -> 多个点
    # y: M x D [60 x 64]  -> 原型
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    # x.unsqueeze(1): Size([N, 1, D])   [300, 1, 64] -> [300, 60, 64] -> 一维 -> 二维
    # y.unsqueeze(0): Size([1, M, D])   [1, 60, 64] -> [300, 60, 64] -> 化成300个点
    #
    x = x.unsqueeze(1).expand(n, m, d)  # Size([N, M, D])
    y = y.unsqueeze(0).expand(n, m, d)  # Size([N, M, D])
    # 求欧式距离
    return torch.pow(x - y, 2).sum(2)   # Size([N, M])
