import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

class CNN_feature_extraction(nn.Module):
    """ 
    CNN feature extraction module 
    we implement a CNN architecture for feature extraction using as reference the one from the atari paper: 
    Playing Atari with Deep Reinforcement Learning. 
    """

    def __init__(self):
        super(CNN_feature_extraction, self).__init__()

        self.layer1 = nn.Sequential(

            #Four chanels input (stacked frames): this are the observations from the environment creating a state st = {xt-3, xt-2, xt-1, xt} where xt is the observation (84x84) at time t 
            #We assume the input image is grayscale that's why we have only one channel per frame, but if the input is RGB we should change the number of input channels to 12 (4 stacked frames x 3 channels)
            #Height and Width of the input image: 84x84 are infered at runtime
            #32 filters, kernel size 8x8, stride 4
            
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # Shape: (N, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Shape: (N, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),# Shape: (N, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(), # Shape: (N, 7*7*64)
            nn.Linear(7*7*64, 512), 
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        return x #encoding vector of size 512 that represents the input state
    
