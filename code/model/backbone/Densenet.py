import torch
import torch.nn as nn
import torchvision
# noinspection PyUnresolvedReferences
from torchsummary import summary

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from attention.CoordAttn_simam import Simam
from attention.MocAttn import MoCAttention
from attention.CGAFusion import CGAFusion
from conv.WTConv2d import WTConv2d

#BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, dropout):
        super(DenseBlock, self).__init__()
        self.dropout = dropout
        self.blockconv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # growth_rate：增长率。一层产生多少个特征图
            nn.Conv2d(in_channels=in_channels, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            WTConv2d(growth_rate, growth_rate, kernel_size=3, stride=1),  # 在3x3 Conv2d之后添加小波卷积
        )
        self.dropout = nn.Dropout(p=self.dropout)
        self.simam = Simam()
 
    def forward(self, x):
        features = self.blockconv(x)
        features = self.simam(features)
        if self.dropout.p > 0:
            features = self.dropout(features)
        return torch.cat([x, features], 1)
 

#BN+1×1Conv+2×2AveragePooling
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, plance):
        super(TransitionBlock, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,out_channels=plance,kernel_size=1,stride=1,padding=0,bias=False),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )
 
    def forward(self, x):
        return self.transition(x)
 
 
class DenseNet(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32, blocks=[6, 12, 24, 16], num_classes=5, dropout = 0.1):
        super(DenseNet, self).__init__()
        bn_size = 4
        self.conv1 = self._create_conv1(in_channels=3, channels=in_channels)
        blocks*4
 
        #第一次执行特征的维度来自于前面的特征提取
        num_features = in_channels
 
        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.layer1 = self._make_layer(num_layers=blocks[0], in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size, dropout=dropout)
        num_features = num_features + blocks[0] * growth_rate
        # 第1个transition 执行 _TransitionLayer（256,128）
        self.transition1 = TransitionBlock(in_channels=num_features, plance=num_features // 2)
        #num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        num_features = num_features // 2
 
        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.layer2 = self._make_layer(num_layers=blocks[1], in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size, dropout=dropout)
        num_features = num_features + blocks[1] * growth_rate
        # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition2 = TransitionBlock(in_channels=num_features, plance=num_features // 2)
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        num_features = num_features // 2
 
        # 第3个DenseBlock有24个DenseLayer, 执行DenseBlock（24,256,32,4）
        self.layer3 = self._make_layer(num_layers=blocks[2], in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size, dropout=dropout)
        num_features = num_features + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（1024,512）
        self.transition3 = TransitionBlock(in_channels=num_features, plance=num_features // 2)
        # num_features减少为原来的一半，执行第3回合之后，第4个DenseBlock的输入的feature应该是：num_features = 512
        num_features = num_features // 2
 
        # 第4个DenseBlock有16个DenseLayer, 执行DenseBlock（16,512,32,4）
        self.layer4 = self._make_layer(num_layers=blocks[3], in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size, dropout=dropout)
        num_features = num_features + blocks[3] * growth_rate
        
        self.attention1 = MoCAttention(28)
        self.attention2 = MoCAttention(14)
        self.attention3 = MoCAttention(7)

        self.cagf1 = CGAFusion(128)
        self.cagf2 = CGAFusion(256)
        
        # # blocks=[6, 12, 64, 48]
        # self.cagf3 = CGAFusion(1152)
        # self.cagf4 = CGAFusion(2688)
        
        # # blocks=[6, 12, 48, 32]
        # self.cagf3 = CGAFusion(896)
        # self.cagf4 = CGAFusion(1920)
        
        # # blocks=[6, 12, 32, 32]
        # self.cagf3 = CGAFusion(640)
        # self.cagf4 = CGAFusion(1664)
        
        # blocks=[6, 12, 24, 16]
        self.cagf3 = CGAFusion(512)
        self.cagf4 = CGAFusion(1024)
 
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, num_classes)
 
    
    def _make_layer(self, num_layers, in_channels, growth_rate, bn_size, dropout):
        # 构造layer，包含多个DenseBlock
        layers = [DenseBlock(in_channels + i * growth_rate, growth_rate, bn_size, dropout) for i in range(num_layers)]
        
        return nn.Sequential(*layers)  # 返回构建的层

    def _create_conv1(self, in_channels, channels, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, x):
        
        x = self.conv1(x)
 
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.attention1(x.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        x = self.cagf1(x, x)
        
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.attention2(x.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        x = self.cagf2(x, x)
        
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.attention3(x.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        x = self.cagf3(x, x)
        
        x = self.layer4(x)
        x = self.cagf4(x, x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

'''
DenseNet121  DenseNet(blocks=[6, 12, 24, 16])
DenseNet169  DenseNet(blocks=[6, 12, 32, 32])
DenseNet201  DenseNet(blocks=[6, 12, 48, 32])
DenseNet264  DenseNet(blocks=[6, 12, 64, 48])
'''
if __name__ == '__main__':
 
    model = DenseNet(blocks=[6, 12, 24, 16]).cuda()
    # print(model)
    input = torch.randn(1, 3, 224, 224).cuda()
    output = model(input)
    print(f"Output shape: {output.shape}")
    # net = DenseNet().cuda()
    # summary(net, (3, 224, 224))