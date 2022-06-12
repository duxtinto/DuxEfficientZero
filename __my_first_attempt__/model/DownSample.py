import torch.nn as nn


# Downsample observations before representation network (See paper appendix Network Architecture)
from model.ResidualBlock import ResidualBlock


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2, momentum=momentum)
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2, momentum=momentum) for _ in range(1)]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, momentum=momentum, stride=2,
                                              downsample=self.conv2)
        self.resblocks2 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x
