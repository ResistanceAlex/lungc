import torch
import torch.nn as nn
import torchvision.models as models

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torch.nn import functional as F

from attention.CoordAttn_simam import CoordAtt
from attention.CoordAttn_simam import Simam
from attention.MocAttn import MoCAttention


# 残差块
class ResidualBlock(nn.Module):
    # 实现子module: Residual Block
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock,self).__init__()
        self.blockconv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace = True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
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
            nn.MaxPool2d(3,2,1)
        )
        # 重复的layer，分别有3,4,6,3个residual block
        self.layer1 = self._make_layer(64,128,3)
        self.layer2 = self._make_layer(128,256,4,stride=2)
        self.layer3 = self._make_layer(256,512,6,stride=2)
        self.layer4 = self._make_layer(512,512,3,stride=2)
        
        self.fc = nn.Identity()
        
        # self.attention = CoordAtt(inp=512,oup=512)
        
        self.attention = MoCAttention(512,256)
        
        self.dropout = nn.Dropout(dropout)
        
        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        # 构造layer，包含多个residual block
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        
        layers=[]
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.pre(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        
        # # 在这里应用注意力机制
        # out = self.attention(out)
        
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 在这里应用注意力机制
        out = self.attention(out)
        
        out = F.avg_pool2d(out,7)
        
        # # 在这里应用注意力机制
        # out = self.attention(out)
        
        out = self.dropout(out)

        # 展平 out
        out = out.view(out.size(0), -1)
        
        # out = self.dropout2(out)

        out = self.fc(out)
        
        return out

if __name__ == "__main__":

    model = LungNoduleClassifier().cuda()  # 将模型移动到CUDA
    dummy_img = torch.randn(1, 3, 224, 224).cuda()  # 假设输入尺寸
    # dummy_bbox = torch.randn(1, 4).cuda()  # 假设边界框
    # output = model(dummy_img, dummy_bbox)  # 测试模型
    output = model(dummy_img)  # 测试模型
    print(f"Output shape: {output.shape}")

    # print(model)