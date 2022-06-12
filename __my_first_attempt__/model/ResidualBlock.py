import torch.nn as nn
from model.utils import conv3x3


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, momentum=0.1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)
        return out
