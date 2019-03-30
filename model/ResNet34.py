#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name : ResNet34.py
File Description : Define my own ResNet34
Author : llw
"""
import torch as t
from torch.nn import functional as F

from model.BasicModel import BasicModel
from config import DEVICE


class BasicBlock(BasicModel):
    """
    Define a basic block of ResNet
    """
    def __init__(self, block_in_channels, block_out_channels, stride=1, shortcut=None):
        """
        __init__() method
        :param block_in_channels: number of channels of input data
        :param block_out_channels: number of channels of output data
        :param stride: stride used in convolutional network
        :param shortcut: shortcut function
        """

        super(BasicBlock, self).__init__()
        self.block_shortcut = shortcut
        self.block_residual = t.nn.Sequential(
            t.nn.Conv2d(in_channels=block_in_channels, out_channels=block_out_channels, kernel_size=3, stride=stride,
                        padding=1),
            t.nn.BatchNorm2d(num_features=block_out_channels),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=block_out_channels, out_channels=block_out_channels, kernel_size=3, stride=1,
                        padding=1),
            t.nn.BatchNorm2d(num_features=block_out_channels)
        )

    def forward(self, x):
        residual = self.block_residual(x)
        shortcut = x if self.block_shortcut is None else self.block_shortcut(x)
        return F.relu(residual + shortcut)


class ResNet34(BasicModel):
    """
    stack all the basic blocks needed and build the complete ResNet
    """

    def __init__(self, in_channels, out_classes=2, shortcut=None):
        super(ResNet34, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.shortcut = shortcut
        self.pre_conv = t.nn.Sequential(
            t.nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(inplace=True),
            t.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ).to(DEVICE)
        self.layer1 = self.make_layer(in_channels=64, out_channels=128, num_of_blocks=3)
        self.layer2 = self.make_layer(in_channels=128, out_channels=256, num_of_blocks=4, stride=2)
        self.layer3 = self.make_layer(in_channels=256, out_channels=512, num_of_blocks=6, stride=2)
        self.layer4 = self.make_layer(in_channels=512, out_channels=512, num_of_blocks=3, stride=2)
        self.out_fc = t.nn.Linear(512, self.out_classes)

    def make_layer(self, in_channels, out_channels, num_of_blocks, stride=1):
        """
        build one part of ResNet
        :param in_channels: same
        :param out_channels: same
        :param num_of_blocks: build how many blocks
        :param stride:  same
        :return:
        """
        assert num_of_blocks >= 2
        shortcut = t.nn.Sequential(
            t.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
            t.nn.BatchNorm2d(num_features=out_channels)
        )
        layers = []
        layers.append(BasicBlock(block_in_channels=in_channels, block_out_channels=out_channels, stride=stride,
                                 shortcut=shortcut))
        for cnt in range(1, num_of_blocks):
            layers.append(BasicBlock(block_in_channels=out_channels, block_out_channels=out_channels))
        return t.nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        y = self.out_fc(x)
        return y