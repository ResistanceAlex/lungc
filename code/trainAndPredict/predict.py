import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from code.model.backbone.LungNoduleClassifier import LungNoduleClassifier
from code.model.backbone.LungNoduleClassifier_b import LungNoduleClassifierResNet50
from dataUtil.LungNoduleDataset import LungNoduleDataset
from torch.utils.data import DataLoader


# 参数设置
result_id = 'result'
model_path = fr'result/{result_id}/pth/best_val_model.pth'  # 替换为你的模型路径
test_image_folder = 'dataset/images/test'
test_label_folder = 'dataset/labels/test'
result_dir = fr'result/{result_id}/test_results'

# 确保结果目录存在
os.makedirs(result_dir, exist_ok=True)

# 读取测试集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),    # 随机垂直翻转
    transforms.ToTensor(),
])

test_dataset = LungNoduleDataset(test_image_folder, test_label_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LungNoduleClassifierResNet50().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

all_test_preds = []
all_test_labels = []
all_test_scores = np.zeros((len(test_dataset), 5))  # 假设有5个类别

# 预测
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        
        all_test_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())
        all_test_scores[i*16:(i+1)*16] = probs.cpu().numpy()

# 绘制ROC曲线
for i in range(5):  # 假设有5个类别
    y_true = (np.array(all_test_labels) == i).astype(int)  # 将标签转换为二元形式
    fpr, tpr, _ = roc_curve(y_true, all_test_scores[:, i])
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
plt.savefig(os.path.join(result_dir, 'roc_curve_final.png'), dpi=300)
plt.close()

# 绘制精确率-召回率曲线
plt.figure(figsize=(10, 5))
for i in range(5):
    y_true = (np.array(all_test_labels) == i).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, all_test_scores[:, i])
    plt.plot(recall, precision, label=f'Class {i}')

# 添加图例和标签
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.savefig(os.path.join(result_dir, 'precision_recall_curve_final.png'), dpi=300)
plt.close()

# 类别混淆矩阵图
conf_matrix = confusion_matrix(all_test_labels, all_test_preds)
plt.figure()
plt.matshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
# 在每个格子中添加计数
for (i, j), val in np.ndenumerate(conf_matrix):
    plt.text(j, i, val, ha='center', va='center', color='white')
plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), dpi=300)
plt.close()

print("result saved.")
