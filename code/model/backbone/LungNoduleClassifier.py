import torch
import torch.nn as nn
import torchvision.models as models

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torch.nn import functional as F

from attention.CoordAttn_simam import Simam
from attention.MocAttn import MoCAttention
from attention.CGAFusion import CGAFusion
from conv.WTConv2d import WTConv2d



# 残差块
class ResidualBlock(nn.Module):
    # 实现子module: Residual Block
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock,self).__init__()
        self.blockconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            WTConv2d(out_channels, out_channels, kernel_size=3, stride=1),  # 在3x3 Conv2d之后添加小波卷积
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = shortcut
        self.simam = Simam()
    
    def forward(self,x):
        out = self.blockconv(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        out = self.simam(out)
        return F.relu(out)

# resnet34 backbone
class LungNoduleClassifier(nn.Module):
    # 实现主module:ResNet34
    # ResNet34包含多个layer，每个layer又包含多个residual block
    # 用子module实现residual block，用_make_layer函数实现layer
    def __init__(self, num_classes=5, dropout=0.1):
        super(LungNoduleClassifier, self).__init__()
        
        # 前几层图像转换
        self.pre=nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            WTConv2d(64, 64, kernel_size=7, stride=1),  # 在Conv2d之后添加小波卷积
            nn.MaxPool2d(3,2,1)
        )
        # 重复的layer，分别有3,4,6,3个residual block
        self.layer1 = self._make_layer(64,128,3)
        self.layer2 = self._make_layer(128,256,4,stride=2)
        self.layer3 = self._make_layer(256,512,6,stride=2)
        self.layer4 = self._make_layer(512,512,3,stride=2)
        
        self.fc = nn.Identity()
        
        self.attention1 = MoCAttention(55)
        self.attention2 = MoCAttention(28)
        self.attention3 = MoCAttention(14)
        self.attention4 = MoCAttention(7)
        self.cagf1 = CGAFusion(128)
        self.cagf2 = CGAFusion(256)
        self.cagf3 = CGAFusion(512)
        
        self.dropout = nn.Dropout(dropout)
        
        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, block_num, stride=1):
        # 构造layer，包含多个residual block
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        layers=[]
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))
        
        for i in range(1,block_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.pre(x)
        
        out = self.layer1(out)
        out = self.attention1(out.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        out = self.cagf1(out, out)
        out = self.layer2(out)
        out = self.attention2(out.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        out = self.cagf2(out, out)    
        out = self.layer3(out)
        out = self.attention3(out.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        out = self.cagf3(out, out)
        out = self.layer4(out)
        out = self.attention4(out.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        out = self.cagf3(out, out)
        

        
        out = F.avg_pool2d(out,7)
        
        # # 在这里应用注意力机制
        # out = self.attention(out)
        
        out = self.dropout(out)

        # 展平 out
        out = out.view(out.size(0), -1)
        
        # out = self.dropout2(out)

        out = self.fc(out)
        
        # 添加 softmax 层
        out = F.softmax(out, dim=1)

        
        return out

if __name__ == "__main__":

    model = LungNoduleClassifier().cuda()  # 将模型移动到CUDA
    dummy_img = torch.randn(1, 3, 224, 224).cuda()  # 假设输入尺寸
    # dummy_bbox = torch.randn(1, 4).cuda()  # 假设边界框
    # output = model(dummy_img, dummy_bbox)  # 测试模型
    output = model(dummy_img)  # 测试模型
    print(f"Output shape: {output.shape}")

    # print(model)