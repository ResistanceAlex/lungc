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
from model.backbone.ViT import Vit
from model.backbone.ViTB import VisionTransformer
from dataUtil.LungNoduleDataset import LungNoduleDataset

from dataUtil.loss import FocalLoss
from dataUtil.loss import LabelSmoothingCrossEntropy

import random
import numpy as np
import torch

seed_value = random.randint(1, 10000)  # 不固定种子
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

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

# 设定训练参数
num_epochs = 4000
best_train_acc = 0.7
best_val_acc = 0.3

def get_unique_result_dir(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir
    else:
        i = 1
        new_dir = f"{base_dir}_{i}"
        while os.path.exists(new_dir):
            i += 1
            new_dir = f"{base_dir}_{i}"
        os.makedirs(new_dir)
        return new_dir

# 使用 get_unique_result_dir 函数获取唯一的结果文件夹
result_dir = get_unique_result_dir('result/result')

os.makedirs(fr'{result_dir}/lung_pic')
os.makedirs(fr'{result_dir}/result_pic')
os.makedirs(fr'{result_dir}/pth')

best_train_model_path = fr'{result_dir}/pth/best_train_model.pth'
best_val_model_path = fr'{result_dir}/pth/best_val_model.pth'
final_lung_nodule_classifier_path = fr'{result_dir}/pth/final_lung_nodule_classifier.pth'

csv_file = os.path.join(result_dir, 'result.csv')

# 创建 CSV 文件并写入表头
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss", "Train Accuracy", "Val Accuracy", "Val Precision", "Val Recall", 
                     "Val F1"])

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

# IMG_SIZE = 224
# IN_CHANNELS = 3
# PATCH_SIZE = 16
# NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 49
# EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS  # 16
# DROPOUT = 0.001

# NUM_HEADS = 8
# ACTIVATION = "gelu"
# NUM_ENCODERS = 4
# NUM_CLASSES = 5
# HIDDEN_LAYER = 768
# model = Vit(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT, NUM_HEADS, ACTIVATION, NUM_ENCODERS,
#                 NUM_CLASSES).to(device)
# model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=16,
#                               depth=16,
#                               num_heads=8,
#                               representation_size=None,
#                               num_classes=5).to(device)
# model = LungNoduleClassifier().to(device)

model = LungNoduleClassifierResNet50().to(device)

# criterion = FocalLoss(alpha=alpha, gamma=2.0)
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingCrossEntropy(smoothing=0.001)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=0, last_epoch=-1, verbose=False)

# 训练过程
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    all_train_preds = []
    all_train_labels = []

    # 使用 tqdm 显示进度条
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (images, labels) in enumerate(train_bar):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_train_preds.extend(predicted.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())
        train_bar.set_postfix(loss=loss.item())

    # 计算指标
    train_accuracy = accuracy_score(all_train_labels, all_train_preds)
    
    # 保存最佳模型
    if train_accuracy > best_train_acc:
        best_train_acc = train_accuracy
        torch.save(model.state_dict(), best_train_model_path)
        print("Best training model saved.")

    # 验证阶段
    
    model.eval()
    all_val_preds = []
    all_val_labels = []
    all_val_scores = np.zeros((len(val_dataset), 5))  # 假设有5个类别

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_val_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
            all_val_scores[i*16:(i+1)*16] = probs.cpu().numpy()

    # 计算指标
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=1)
    val_recall = recall_score(all_val_labels, all_val_preds, average='macro')
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_val_labels, all_val_preds, labels=[0, 1]).ravel()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Train_Accuracy: {train_accuracy:.4f}, Val_Accuracy: {val_accuracy:.4f}, Val_Precision: {val_precision:.4f}, "
          f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    
    # 将结果写入 CSV 文件
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, running_loss/len(train_loader), train_accuracy, 
                         val_accuracy, val_precision, val_recall, val_f1])

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), best_val_model_path)
        print("Best validation model saved.")
    
    scheduler.step()
    optimizer.step()

# 绘制ROC曲线
for i in range(5):  # 假设有5个类别
    y_true = (np.array(all_val_labels) == i).astype(int)  # 将标签转换为二元形式
    fpr, tpr, _ = roc_curve(y_true, all_val_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

def printPic():
    # 添加图例和标签
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(result_dir, 'result_pic/roc_curve_final.png'), dpi=300)
    plt.close()

    # 绘制精确率-召回率曲线
    plt.figure(figsize=(10, 5))
    for i in range(5):
        y_true = (np.array(all_val_labels) == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, all_val_scores[:, i])
        plt.plot(recall, precision, label=f'Class {i}')

    # 添加图例和标签
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(result_dir, 'result_pic/precision_recall_curve_final.png'), dpi=300)
    plt.close()

    # 类别混淆矩阵图
    conf_matrix = confusion_matrix(all_val_labels, all_val_preds)
    plt.figure()
    plt.matshow(conf_matrix, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # 在每个格子中添加计数
    for (i, j), val in np.ndenumerate(conf_matrix):
        plt.text(j, i, val, ha='center', va='center', color='white')
    plt.savefig(os.path.join(result_dir, f'result_pic/confusion_matrix_epoch_{num_epochs}.png'), dpi=300)
    plt.close()

printPic()
# 保存最终模型
torch.save(model.state_dict(), final_lung_nodule_classifier_path)
print("Final model saved.")
