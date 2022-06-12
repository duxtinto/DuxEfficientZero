import numpy as np
import torch.nn as nn


# Encode the observations into hidden states
from model.DownSample import DownSample
from model.ResidualBlock import ResidualBlock
from model.utils import conv3x3


class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_channels,
        downsample,
        momentum=0.1,
    ):
        """Representation network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
            )
        self.conv = conv3x3(
            observation_shape[0],
            num_channels,
        )
        self.bn = nn.BatchNorm2d(num_channels, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean
