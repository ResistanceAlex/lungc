import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import random
import time
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
from dataUtil.LungNoduleDataset import LungNoduleDataset

from dataUtil.loss import FocalLoss
from dataUtil.loss import LabelSmoothingCrossEntropy

from predict import predict
from drawPic.heatmap import draw_heatmap
from drawPic.result_pic import draw_pic

# 设定训练种子
def seed_everything(seed):   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 生成结果文件夹
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

# 绘图
def printPic(all_val_labels, all_val_scores, all_val_preds, result_dir, num_epochs):
    
    # 绘制ROC曲线
    for i in range(5):  # 假设有5个类别
        y_true = (np.array(all_val_labels) == i).astype(int)  # 将标签转换为二元形式
        fpr, tpr, _ = roc_curve(y_true, all_val_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
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

# 训练
def train(num_epochs, best_train_acc, best_val_acc, 
          model, criterion, optimizer, scheduler, heatmap, 
          train_dataset, val_dataset, train_loader, val_loader):

    # 记录训练开始时间
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # 不固定种子
    seed = random.randint(1, 10000)
    seed_everything(seed)

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

    # 设定训练参数
    num_epochs = num_epochs
    best_train_acc = best_train_acc
    best_val_acc = best_val_acc

    # 设置模型和超参
    model = model
    criterion = criterion
    optimizer = optimizer
    scheduler = scheduler
    
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

    # 保存最终模型
    torch.save(model.state_dict(), final_lung_nodule_classifier_path)
    print("Final model saved.")

    draw_pic(result_id=result_dir)
    predict(result_id=result_dir,model=model)
    if(heatmap!=None):
        draw_heatmap(result_id=result_dir,model=model)
    
    printPic(all_val_labels, all_val_scores, all_val_preds, result_dir, num_epochs)

    print("All Pic saved.")
    
    # 训练结束后计算并输出时长
    end_time = time.time()
    elapsed_time = end_time - start_time

    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60

    print(f"Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
