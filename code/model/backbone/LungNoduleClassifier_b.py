import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from attention.CoordAttn_simam import Simam
from attention.MocAttn import MoCAttention
from conv.WTConv2d import WTConv2d


# ResNet-50/101/152 残差结构 ResNetblock
class ResNetblock(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetblock, self).__init__()
        self.blockconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            WTConv2d(out_channels, out_channels, kernel_size=3, stride=1),  # 在3x3 Conv2d之后添加小波卷积
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels * 4)
        )
        self.simam = Simam()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        residual = x
        out = self.blockconv(x)
        if hasattr(self, 'shortcut'):  # 如果self中含有shortcut属性
            residual = self.shortcut(x)
        out += residual
        out = self.simam(out)
        return F.relu(out)


class LungNoduleClassifierResNet50(nn.Module):
    def __init__(self, block=ResNetblock, num_classes=5, dropout=0.001):
        super(LungNoduleClassifierResNet50, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            WTConv2d(64, 64, kernel_size=7, stride=1),  # 在Conv2d之后添加小波卷积
            nn.MaxPool2d((3, 3), stride=2)
        )
        self.in_channels = 64
        # ResNet50中的四大层，每大层都是由ConvBlock与IdentityBlock堆叠而成
        self.layer1 = self.make_layer(ResNetblock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResNetblock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResNetblock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResNetblock, 512, 3, stride=2)
        
        self.attention1 = MoCAttention(55)
        self.attention2 = MoCAttention(28)
        self.attention3 = MoCAttention(14)
        self.attention4 = MoCAttention(7)

        self.avgpool = nn.AvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 4, num_classes)
        self.dropout = nn.Dropout(dropout)

    # 每个大层的定义函数
    def make_layer(self, block, channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.layer1(out)
        out = self.attention1(out.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        out = self.layer2(out)
        out = self.attention2(out.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        out = self.layer3(out)
        out = self.attention3(out.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        out = self.layer4(out)
        out = self.attention4(out.permute(2, 3, 1, 0)).permute(3, 2, 0, 1)
        
        out = F.avg_pool2d(out,7)
        
        out = self.dropout(out)

        # 展平 out
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        
        return out

if __name__ == "__main__":

    model = LungNoduleClassifierResNet50().cuda()  # 将模型移动到CUDA
    dummy_img = torch.randn(1, 3, 224, 224).cuda()  # 假设输入尺寸
    
    output = model(dummy_img)  # 测试模型
    print(f"Output shape: {output.shape}")
