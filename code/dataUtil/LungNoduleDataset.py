import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

class LungNoduleDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        label_name = os.path.join(self.label_folder, self.image_files[idx].replace('.jpg', '.txt'))

        # 读取图像并转换为NumPy数组，准备进行裁剪
        image = Image.open(img_name).convert('RGB')
        image = np.array(image)  # 转换为NumPy数组进行边界框裁剪
        
        # 读取标签
        with open(label_name, 'r') as f:
            lines = f.readlines()
        
        # 假设每个文件只有一行（根据你的需求，可以修改）
        label_info = lines[0].strip().split(' ')
        label = int(label_info[0]) - 1  # 假设标签为 1~5，转换为 0~4

        # 获取边界框信息并转换为Tensor
        bbox = torch.tensor([float(label_info[1]), float(label_info[2]), 
                         float(label_info[3]), float(label_info[4])], dtype=torch.float32)

        # 裁剪图像
        cropped_image = self.crop_image_by_bbox(image, bbox.tolist())  # 裁剪时使用列表

        # 转换为PIL格式
        cropped_image = Image.fromarray(cropped_image)

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, label

    def crop_image_by_bbox(self, image, bbox):
        """
        根据边界框信息裁剪图像
        bbox: [x_center, y_center, width, height], 值为归一化后的坐标
        """
        h, w, _ = image.shape  # 获取图像尺寸

        # 将边界框信息从归一化转换为像素坐标
        x_center = bbox[0] * w
        y_center = bbox[1] * h
        width = bbox[2] * w
        height = bbox[3] * h

        # 计算左上角和右下角的像素坐标，扩展10像素
        top_left_x = int(x_center - width / 2) - 10
        top_left_y = int(y_center - height / 2) - 10
        bottom_right_x = int(x_center + width / 2) + 10
        bottom_right_y = int(y_center + height / 2) + 10

        # 裁剪图像
        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        return cropped_image



if __name__ == "__main__":
    
    # 数据增强和转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 适应网络输入大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),    # 随机垂直翻转
        transforms.ToTensor(),
    ])
    
    # 创建数据集和数据加载器
    image_folder = 'dataset/images/train'  # 替换为你的路径
    label_folder = 'dataset/labels/train'  # 替换为你的路径
    dataset = LungNoduleDataset(image_folder, label_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 统计类别数量
    class_count = {}
    
    for _, _, labels in dataloader:
        for label in labels:
            label_item = label.item()  # 获取标签的整数值
            if label_item in class_count:
                class_count[label_item] += 1
            else:
                class_count[label_item] = 1

    # 输出类别统计
    for cls, count in class_count.items():
        print(f'Class {cls + 1}: {count} samples')
