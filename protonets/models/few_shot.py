import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from protonets.models import register_model
from .utils import euclidean_dist


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # 展开为二维张量
        return x.view(x.size(0), -1)


# 这里是全文核心 网络架构都在这里了
class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder

    # 计算错误率
    def loss(self, sample):
        # print("sample", sample['xs'].size())
        # torch.Size([60, 5, 1, 28, 28])
        # 意思是60类 每类5个样本 每个样本为28 * 28的图
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # query

        n_class = xs.size(0)  # N类 -> 60
        assert xq.size(0) == n_class  # Support Set 和 Query Set的类总数需要保持一致
        n_support = xs.size(1)  # 多少个Support Example -> 5
        n_query = xq.size(1)  # 多少个Query Example -> 5

        # 每个example的label，也就是对应的分类
        # torch.arange(0, n_class) : [0, 1, 2, 3, 4, ... ,n_class - 1] -> Size([n_class])
        # torch.view(n_class, 1, 1): [ [[0]], [[1]], [[2]], ..., [[n_class - 1]]] -> Size([n_class, 1, 1])
        # torch.expand(n_class, n_query, 1): [ [[0] * n_query], [[1] * n_query], ..., [[n_class - 1] * n_query]]
        #   不扩展内存只是指针调用来扩展张量
        # 因为我们是按顺序来的[60, 5, 1, 28, 28] -> 60类 每类5个example  每个example是[1, 28, 28]的图
        # 打标的时候应该是每个图对应一种，我们直接按index打标签，也就是第一类的为0，第二类为1...
        # 即输出应该为 [60, 5, 60] 每个图输出60个数字代表每种类型的概率，取最大就是我们的预测值
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        # torch.Size([60, 5, 1]) 代表60类 每类5个 每个代表同一个类别
        target_inds = Variable(target_inds, requires_grad=False)  # 不计算梯度

        # 显卡判断
        if xq.is_cuda:
            target_inds = target_inds.cuda()

        # 将Support Set 和 Query Set中的图片直接扩展出来
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)  # 安装维度0来拼接
        # print(x.size()) -> torch.Size([600, 1, 28, 28])

        # 前向传输 也就是进入网络处理
        # 新版本中无需调用forward，实例化的时候就会自动去调用
        z = self.encoder.forward(x)
        # print('z', z.size()) -> torch.Size([600, 64])

        z_dim = z.size(-1)  # 64
        # n_class * n_support = 60 * 5 = 300
        # z[:n_class * n_support].view(n_class, n_support, z_dim): Size([60, 5, 64])
        # z_proto: Size([60, 64])
        # 下面的z_proto就是原型的位置！！！也就是对矢量取平均得到的中心位置！
        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)  # 在轴为1上进行取平均 Size([60, 64])
        # 下面是Query Set处理后的数据
        zq = z[n_class * n_support:]  # Query Set处理后的 Size([300, 64])  -> 300 个Example 每个为64

        # 求出欧式距离
        dists = euclidean_dist(zq, z_proto)  # Size([300, 60])

        # log(softmax(x))
        # F.log_softmax(-dists, dim=1).size() -> Size([300, 60])
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)  # Size([60, 5 , 60])
        # log_p_y的意思是有60类 每类5张 每张对应的60种的概率

        # target_inds -> Size([60, 5, 1])
        # -log_p_y.gather(2, target_inds) -> Size([60 , 5, 1])
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # 在2轴上取最大值的索引y_hat，也就是属于哪一类
        _, y_hat = log_p_y.max(2)
        # 然后和target_inds比较，如果相等就是预测对了
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    # 加载模型 返回加载对应参数的模型
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # 数据归一 使数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2)  # 池化
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()  # 展开为二维张量
    )  # 通过Sequential把前向传输过程给省略了
    # encoder在这里就是这个网络的结构 它用了非一般写法来适应动态参数问题 可以改写成一般写法
    return Protonet(encoder)
