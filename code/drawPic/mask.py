import argparse
import cv2
import numpy as np
import os

def read_labels(label_path):
    """
    读取标签文件，返回标签和边界框信息
    """
    labels = []
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            labels.append(label)
            boxes.append((x_center, y_center, width, height))
    return labels, boxes

def main():
    images_bg = r'dataset/images/train/707.jpg'
    labels_path = r'dataset/labels/train/707.txt'
    output_image_path = r'with_boxes_707.jpg'

    # 加载原始输入图像，并展示
    image = cv2.imread(images_bg)
    if image is None:
        raise ValueError(f"Image not found: {images_bg}")

    # 读取标签
    labels, boxes = read_labels(labels_path)

    # 绘制边界框
    for (x_center, y_center, width, height) in boxes:
        x_center *= image.shape[1]
        y_center *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]

        # 计算边界框的左上角和右下角坐标，扩展10像素
        top_left = (int(x_center - width / 2) - 10, int(y_center - height / 2) - 10)
        bottom_right = (int(x_center + width / 2) + 10, int(y_center + height / 2) + 10)

        # 绘制矩形框
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)



    # 保存生成的图像
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved to: {output_image_path}")
    
    
    # # 设置显示窗口大小
    # window_name = "Lung Nodules"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, 800, 800)  # 调整窗口大小

    # cv2.imshow(window_name, image)
    # cv2.waitKey()

if __name__ == "__main__":
    main()
