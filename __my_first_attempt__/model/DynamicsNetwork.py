import torch.nn as nn
import numpy as np


# Predict next hidden states given current states and actions
from model.ResidualBlock import ResidualBlock
from model.utils import conv3x3


class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        """Dynamics network
        Parameters
        ----------
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        block_output_size_reward: int
            dim of flatten hidden states
        lstm_hidden_size: int
            dim of lstm hidden
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        """
        super().__init__()
        self.num_channels = num_channels
        self.lstm_hidden_size = lstm_hidden_size

        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = nn.BatchNorm2d(num_channels - 1, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.reward_resblocks = nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - 1, reduced_channels_reward, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels_reward, momentum=momentum)
        self.block_output_size_reward = block_output_size_reward
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x, reward_hidden):
        state = x[:,:-1,:,:]
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)

        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

        for block in self.resblocks:
            for name, param in block.named_parameters():
                dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean
