import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, upsample=None, instance_norm=True, relu=True):
    """
    customized convolutional layer
    :param in_channels: input dimension
    :param out_channels: output dimension
    :param kernel_size: convolutional kernel
    :param stride: stride
    :param upsample: the scale size if needs upsample before convolution operation
    :param instance_norm: batch normalization layer
    :param relu: activation function
    :return: each layer of this block
    """
    layers = []
    if upsample: # TODO
        layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))
        # layers.append(F.interpolate())
    layers.append(nn.ReflectionPad2d(kernel_size // 2))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU())

    return layers


class VGG(nn.Module):

    def __init__(self, features):
        """
        initialization
        :param features: the module container
        """
        super().__init__()
        self.features = features
        
        # 我们要提取的层是第 3、8、15、22 层（第一、二、三、四次池化前卷积 ReLU 的结果）
        # 这四层分别反映了不同层次的特征提取，在风格重构中它们都有用。
        self.layer_name_dict = {
            '3': "relu_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3",
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        :return: the specified feature of the output from intermediate layers
        """
        outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_dict:
                outputs.append(x)
        return outputs

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1),
            *ConvLayer(channels, channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.conv(x) + x


class TransformNet(nn.Module):

    def __init__(self, base):
        super().__init__()
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, kernel_size=9), # TODO item
            *ConvLayer(base, base*2, kernel_size=3, stride=2),
            *ConvLayer(base*2, base*4, kernel_size=3, stride=2),
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(base*4),
            ResidualBlock(base*4),
            ResidualBlock(base*4),
            ResidualBlock(base*4),
            ResidualBlock(base*4)]
        )
        # self.residuals = nn.Sequential(*[ResidualBlock(base * 4) for i in range(5)])

        self.upsampling = nn.Sequential(
            *ConvLayer(base*4, base*2, kernel_size=3, upsample=2),
            *ConvLayer(base*2, base, kernel_size=3, upsample=2),
            *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False),
        )

    def forward(self, x):
        x = self.downsampling(x)
        x = self.residuals(x)
        x = self.upsampling(x)
        return x
