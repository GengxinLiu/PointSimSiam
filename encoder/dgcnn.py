import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # n x n
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, extra_dim=False):
    batch_size, num_dims, num_points = x.shape
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if extra_dim is False:
            idx = knn(x, k=k)  # b, n, k
        else:
            idx = knn(x[:, 6:], k=k)  # idx = knn(x[:, :3], k=k)

    device = torch.device(x.device)
    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx += idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature  # (batch_size, 2 * num_dims, num_points, k)


class DGCNN(nn.Module):
    def __init__(self, channel=3, multi=1., k=20, output_dim=1024):
        super(DGCNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(int(multi * 64))
        self.bn2 = nn.BatchNorm2d(int(multi * 64))
        self.bn3 = nn.BatchNorm2d(int(multi * 128))
        self.bn4 = nn.BatchNorm2d(int(multi * 256))
        self.bn5 = nn.BatchNorm1d(int(multi * output_dim))

        self.conv1 = nn.Sequential(nn.Conv2d(channel * 2, int(multi * 64), kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(int(multi * 64) * 2, int(multi * 64), kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(int(multi * 64) * 2, int(multi * 128), kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(int(multi * 128) * 2, int(multi * 256), kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(int(multi * 256) * 2, int(multi * output_dim), kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.k = k
        self.output_dim = int(multi * output_dim)

    def forward(self, x, global_feature=True):
        """
        :param x: (b, 3, N)
        :param global_feature: return gloabl feature (b, output_dim) or point feature (b, N, output_dim)
        """
        batch_size = x.size()[0]
        x = get_graph_feature(x, k=self.k)  # 每个点的特征和它k个邻居点的特征的差 拼接 自身特征 [x-xk || x]
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        if global_feature:
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            # x = torch.cat((x1, x2), 1)
            return x1
        else:
            return x.permute(0, 2, 1)


if __name__ == '__main__':
    dgcnn = DGCNN(multi=1, k=20, output_dim=1024)
    total = sum([param.nelement() for param in dgcnn.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))  # 0.62M

    # from thop import profile
    # input = torch.randn(1, 3, 1024)
    # flops, params = profile(dgcnn, inputs=(input,))
    # print('FLOPs {:.3f}G'.format(flops / 10e9), params)
    # points = torch.rand(2, 3, 2048).float()
    # print(points.size())
    # features = dgcnn(points, global_feature=True)
    # print(features.shape)
