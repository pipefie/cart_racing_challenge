import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class CNN_feature_extraction(BaseFeaturesExtractor):
    """ 
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    This corresponds to the number of unit for the last layer.

    CNN feature extraction module 
    we implement a CNN architecture for feature extraction using as reference the one from the atari paper: 
    Playing Atari with Deep Reinforcement Learning. 
    """

    def __init__(self,  observation_space: spaces.Box, features_dim: int = 512):
        # We will set self.features_dim ourselves after building the network.
        super().__init__(observation_space, features_dim)
        # Expect CHW because VecTransposeImage('first') is applied to the VecEnv
        c, h, w = observation_space.shape  # e.g., (4, 84, 84) for gray×4 stacked frames


        self.conv = nn.Sequential(

            #Four chanels input (stacked frames): this are the observations from the environment creating a state st = {xt-3, xt-2, xt-1, xt} where xt is the observation (84x84) at time t 
            #We assume the input image is grayscale that's why we have only one channel per frame, but if the input is RGB we should change the number of input channels to 12 (4 stacked frames x 3 channels)
            #Height and Width of the input image: 84x84 are infered at runtime
            #32 filters, kernel size 8x8, stride 4
            
            nn.Conv2d(c, 32, kernel_size=8, stride=4), # Shape: (N, 32, 20, 20)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Shape: (N, 64, 9, 9)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),# Shape: (N, 64, 7, 7)
            nn.ReLU(inplace=True),
            nn.Flatten(), # Shape: (N, 7*7*64)
        )

        # Compute conv output size dynamically with a dummy pass
        with th.no_grad():
            n_flatten = self.conv(th.zeros(1, c, h, w)).view(1, -1).shape[1]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 512),
            nn.ReLU(inplace=True),
        )

        # Tell SB3 the final number of features
        self.features_dim = 512

    def forward(self, observations: th.Tensor) -> th.Tensor:

        x = self.conv(observations)
        x = self.linear(x)            # -> (N, 512)
        return x
    
