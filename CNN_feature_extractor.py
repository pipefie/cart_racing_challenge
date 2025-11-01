import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class CNN_feature_extraction(BaseFeaturesExtractor):
    """
    Espera observaciones CHW uint8 apiladas por canales (C = base_channels * N_STACK).
    Convierte a float y escala a [0,1] internamente.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Tras VecFrameStack(HWC) + VecTransposeImage -> (C, H, W)
        c, h, w = observation_space.shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),  # -> (N, 32, 20, 20) si 84x84
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (N, 64, 9, 9)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (N, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),                                # -> (N, 7*7*64)
        )

        with th.no_grad():
            n_flatten = self.conv(th.zeros(1, c, h, w)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )

        # SB3 expone features_dim como property (sin setter); usar el protegido
        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # uint8 [0,255] -> float32 [0,1]
        x = observations.float() / 255.0
        x = self.conv(x)
        x = self.linear(x)
        return x
