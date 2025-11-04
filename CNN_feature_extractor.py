from __future__ import annotations

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """Nature-CNN style encoder that expects stacked CHW uint8 observations."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        channels, height, width = observation_space.shape

        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, channels, height, width)
            n_flatten = self.conv(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )

        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float() / 255.0
        x = self.conv(x)
        x = self.linear(x)
        return x


# Backwards compatibility alias for previous class name.
CNN_feature_extraction = CNNFeatureExtractor
