import gin
import MinkowskiEngine as ME
import torch
import torch.nn as nn

from co3d_3d.src.models.mink.base_model import MinkowskiBaseModel


class GlobalMaxAvgPool(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, tensor):
        x = self.global_max_pool(tensor)
        y = self.global_avg_pool(tensor)
        return ME.cat(x, y)


@gin.configurable
class MinkowskiFCNN(MinkowskiBaseModel):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3,
    ):
        MinkowskiBaseModel.__init__(self, D)

        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=kernel_size,
            D=D,
        )
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.mlp1 = self.get_mlp_block(in_channel, channels[0])
        self.conv1 = self.get_conv_block(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = self.get_conv_block(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv3 = self.get_conv_block(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv4 = self.get_conv_block(
            channels[3],
            channels[4],
            kernel_size=kernel_size,
            stride=2,
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(
                channels[1] + channels[2] + channels[3] + channels[4],
                embedding_channel // 4,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 4,
                embedding_channel // 2,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 2,
                embedding_channel,
                kernel_size=3,
                stride=2,
            ),
        )

        self.max_pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.final = nn.Sequential(
            GlobalMaxAvgPool(),
            self.get_mlp_block(embedding_channel * 2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

        # No, Dropout, last 256 linear, AVG_POOLING 92%

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.TensorField):
        x = self.mlp1(x)
        y = x.sparse()

        y = self.conv1(y)
        y1 = self.max_pool(y)

        y = self.conv2(y1)
        y2 = self.max_pool(y)

        y = self.conv3(y2)
        y3 = self.max_pool(y)

        y = self.conv4(y3)
        y4 = self.max_pool(y)

        x1 = y1.slice(x)
        x2 = y2.slice(x)
        x3 = y3.slice(x)
        x4 = y4.slice(x)

        x = ME.cat(x1, x2, x3, x4)

        y = self.conv5(x.sparse())
        return self.final(y).F


@gin.configurable
class MinkowskiSplatFCNN(MinkowskiFCNN):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3,
    ):
        MinkowskiFCNN.__init__(
            self, in_channel, out_channel, kernel_size, embedding_channel, channels, D
        )

    def forward(self, x: ME.TensorField):
        x = self.mlp1(x)
        y = x.splat()

        y = self.conv1(y)
        y1 = self.max_pool(y)

        y = self.conv2(y1)
        y2 = self.max_pool(y)

        y = self.conv3(y2)
        y3 = self.max_pool(y)

        y = self.conv4(y3)
        y4 = self.max_pool(y)

        x1 = y1.interpolate(x)
        x2 = y2.interpolate(x)
        x3 = y3.interpolate(x)
        x4 = y4.interpolate(x)

        x = ME.cat(x1, x2, x3, x4)
        y = self.conv5(x.sparse())

        return self.final(y).F
