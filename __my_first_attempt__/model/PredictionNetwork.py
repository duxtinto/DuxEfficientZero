import torch.nn as nn

from model.ResidualBlock import ResidualBlock
from model.utils import mlp


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
        momentum=0.1,
        init_zero=False,
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        block_output_size_value: int
            dim of flatten hidden states
        block_output_size_policy: int
            dim of flatten hidden states
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        """
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.bn_value = nn.BatchNorm2d(reduced_channels_value, momentum=momentum)
        self.bn_policy = nn.BatchNorm2d(reduced_channels_policy, momentum=momentum)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(self.block_output_size_value, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum)
        self.fc_policy = mlp(self.block_output_size_policy, fc_policy_layers, action_space_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        value = self.bn_value(value)
        value = nn.functional.relu(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = nn.functional.relu(policy)

        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value
