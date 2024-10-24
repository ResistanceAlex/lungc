import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score)
from sklearn.preprocessing import label_binarize

from model.backbone.LungNoduleClassifier import LungNoduleClassifier
from model.backbone.LungNoduleClassifier_b import LungNoduleClassifierResNet50
from model.backbone.ViTB import VisionTransformer
from model.backbone.Densenet import DenseNet

from dataUtil.LungNoduleDataset import LungNoduleDataset
from dataUtil.loss import FocalLoss
from dataUtil.loss import LabelSmoothingCrossEntropy
from train import train

import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强和转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),    # 随机垂直翻转
    transforms.RandomRotation(15),      # 随机旋转15度
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
image_folder = 'dataset/images/train'
label_folder = 'dataset/labels/train'
val_image_folder = 'dataset/images/val'
val_label_folder = 'dataset/labels/val'

train_dataset = LungNoduleDataset(image_folder, label_folder, transform=transform)
val_dataset = LungNoduleDataset(val_image_folder, val_label_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# 计算权重
class_counts = {
    1: 463,
    2: 820,
    3: 1197,
    4: 477,
    5: 362
}

total_count = sum(class_counts.values())
weights = {k: total_count / (len(class_counts) * v) for k, v in class_counts.items()}

# 将权重转换为Tensor
alpha = torch.tensor([weights[1], weights[2], weights[3], weights[4], weights[5]], dtype=torch.float32)
alpha = alpha.to(device)

# 初始化模型、损失函数和优化器
# 使用vit之前要将热力图heatmap = None
# File "/home/lab501-2/lungc/code/drawPic/heatmap.py", line 66, in generate_heatmap
#     final_conv_layer = model.layer4[1].blockconv[1]  # 访问最后一个卷积层
#   File "/home/lab501-2/.conda/envs/dlnet/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1269, in __getattr__
#     raise AttributeError("'{}' object has no attribute '{}'".format(
# AttributeError: 'VisionTransformer' object has no attribute 'layer4'

heatmap = 1

# model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=16,
#                               depth=16,
#                               num_heads=8,
#                               representation_size=None,
#                               num_classes=5).to(device)
# model = LungNoduleClassifier().to(device)
# model = LungNoduleClassifierResNet50().to(device)

model = DenseNet(blocks=[6, 12, 24, 16]).to(device)
# model = DenseNet(blocks=[6, 12, 64, 48]).to(device)

# 设定训练参数
num_epochs = 300
best_train_acc = 0.1
best_val_acc = 0.1

criterion = FocalLoss(alpha=alpha, gamma=1.5)
# criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingCrossEntropy(smoothing=0.001)

optimizer = optim.AdamW(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64, eta_min=0, last_epoch=-1, verbose=False)

# 训练并绘图
train(num_epochs, 
      best_train_acc, 
      best_val_acc, 
      model, 
      criterion, 
      optimizer, 
      scheduler, 
      heatmap,
      train_dataset,
      val_dataset,
      train_loader,
      val_loader)