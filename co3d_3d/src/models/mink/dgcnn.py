import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device("cuda")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    x = x.transpose(2, 1).contiguous()
    # batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN_cls(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.emb_dims = emb_dims
        self.dropout = dropout
        self.k = k

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def convblock_max(self, x: torch.Tensor, k: int, conv: nn.Module):
        x = get_graph_feature(x, k)
        x = conv(x)
        x_max = x.max(dim=-1, keepdim=False)[0]
        return x_max, x

    def forward(self, x):
        batch_size = x.size(0)
        x1, x = self.convblock_max(x, self.k, self.conv1)
        # x1 = (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x = (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)

        x2, x = self.convblock_max(x1, self.k, self.conv2)
        # x2 = (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x = (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)

        x3, x = self.convblock_max(x2, self.k, self.conv3)
        # x3 = (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        # x = (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)

        x4, x = self.convblock_max(x3, self.k, self.conv4)
        # x4 = (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        # x = (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)
        # x = (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)
        # (batch_size, 256) -> (batch_size, output_channels)
        return x


class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k

        self.conv1 = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv6 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def convblock_max(
        self, x: torch.Tensor, k: int, conv: nn.Module, dim9: bool = False
    ):
        x = get_graph_feature(x, k, dim9)
        x = conv(x)
        x_max = x.max(dim=-1, keepdim=False)[0]
        return x_max, x

    def forward(self, x):
        num_points = x.size(2)

        # (batch_size, 9, num_points)
        x1, x = self.convblock_max(x, self.k, self.conv1, dim9=True)
        # x = (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        # x1 = (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x2, x = self.convblock_max(x1, self.k, self.conv2)
        # x = (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        # x2 = (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x3, x = self.convblock_max(x2, self.k, self.conv3)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        # x3 (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv4(x)
        # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv5(x)
        # (batch_size, 1024+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv6(x)
        # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x


class TestDGCNN(unittest.TestCase):
    def setUp(self):
        self.dgcnn = DGCNN_cls().cuda()

    def test_forward(self):
        B, C, N = 4, 3, 2000
        input = torch.rand(B, C, N).cuda()
        self.dgcnn(input)
