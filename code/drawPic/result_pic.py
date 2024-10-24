import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve)

def plot_loss_curve(data, result_dir):
    plt.figure()
    plt.plot(data['Epoch'], data['Loss'], color='red', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True) 
    plt.savefig(os.path.join(result_dir, 'result_pic/loss_curve.png'), dpi=300)
    plt.close()

def plot_f1_curve(data, result_dir):
    plt.figure()
    plt.plot(data['Epoch'], data['Val F1'], color='blue', lw=2, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'result_pic/f1_curve.png'), dpi=300)
    plt.close()

def plot_accuracy_curve(data, result_dir):
    plt.figure()
    plt.plot(data['Epoch'], data['Train Accuracy'], color='blue', lw=2, label='Train Accuracy')
    plt.plot(data['Epoch'], data['Val Accuracy'], color='green', lw=2, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True) 
    plt.savefig(os.path.join(result_dir, 'result_pic/accuracy_curve.png'), dpi=300)
    plt.close()

def draw_pic(result_id):
    # 定义结果文件夹和CSV文件路径
    result_dir = result_id  # 结果文件夹路径
    csv_file = os.path.join(result_dir, 'result.csv')

    # 读取 CSV 文件
    data = pd.read_csv(csv_file)

    # 调用绘图方法
    plot_loss_curve(data, result_dir)
    plot_f1_curve(data, result_dir)
    plot_accuracy_curve(data, result_dir)
    
    print("result saved.")

if __name__ == "__main__":
    result_id = 'result/result_1'
    draw_pic(result_id)