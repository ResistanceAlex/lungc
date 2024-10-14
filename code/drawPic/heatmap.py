import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from code.model.backbone.LungNoduleClassifier import LungNoduleClassifier

from code.model.backbone.LungNoduleClassifier_b import LungNoduleClassifierResNet50

def read_labels(label_path):
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

def crop_and_resize(image, bbox, o_image_path):
    x_center, y_center, width, height = bbox
    x_center *= image.shape[1]
    y_center *= image.shape[0]
    width *= image.shape[1]
    height *= image.shape[0]

    # 计算左上角和右下角坐标，扩展10像素
    top_left = (int(x_center - width / 2) - 10, int(y_center - height / 2) - 10)
    bottom_right = (int(x_center + width / 2) + 10, int(y_center + height / 2) + 10)

    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    resized_cropped_image = cv2.resize(cropped_image, (224, 224))  # 改为模型的输入尺寸
    cv2.imwrite(o_image_path, resized_cropped_image)
    
    return resized_cropped_image

def generate_heatmap(model, image, target_class):
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device).unsqueeze(0)

    global gradients, activation
    gradients = None
    activation = None

    def get_gradients(module, grad_input, grad_output):
        global gradients
        gradients = grad_output[0]

    def get_activation(module, input, output):
        global activation
        activation = output

    final_conv_layer = model.layer4[1].blockconv[1]  # 访问最后一个卷积层
    final_conv_layer.register_forward_hook(get_activation)
    final_conv_layer.register_backward_hook(get_gradients)

    output = model(image)

    model.zero_grad()
    one_hot_output = torch.zeros((1, output.size(-1)), dtype=torch.float32).to(device)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output)

    weights = F.adaptive_avg_pool2d(gradients, 1)
    cam = torch.zeros(activation.shape[2:], dtype=torch.float32).to(device)

    for i in range(weights.size(1)):
        cam += weights[0][i] * activation[0][i]

    cam = F.relu(cam)
    cam = cam.cpu().detach().numpy()
    cam -= cam.min()
    cam /= cam.max()

    return cam

def overlay_heatmap(heatmap, original_image):
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.clip(heatmap, 0, 255)
    original_image_pil = transforms.ToPILImage()(original_image)
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(original_image_pil.size))

    heatmap_color = plt.get_cmap('jet')(heatmap_resized)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    overlayed_image = np.array(original_image_pil) * 0.5 + heatmap_color * 0.5
    return overlayed_image.astype(np.uint8)

def main(image_id, result_path):
    pth_path = fr'{result_path}/pth/best_train_model.pth'
    images_bg = fr'dataset/images/train/{image_id}.jpg'
    labels_path = fr'dataset/labels/train/{image_id}.txt'
    os.makedirs(fr'{result_path}/lung_pic/{image_id}/', exist_ok=True)
    output_image_path = fr'{result_path}/lung_pic/{image_id}/overlayed_heatmap_{image_id}.jpg'
    o_image_path = fr'{result_path}/lung_pic/{image_id}/overlayed_o_{image_id}.jpg'

    image = cv2.imread(images_bg)
    if image is None:
        raise ValueError(f"Image not found: {images_bg}")

    labels, boxes = read_labels(labels_path)
    bbox = boxes[0]
    target_class = labels[0] - 1

    cropped_resized_image = crop_and_resize(image, bbox, o_image_path)

    transform = transforms.Compose([transforms.ToTensor()])
    cropped_image_tensor = transform(cropped_resized_image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LungNoduleClassifier().to(device)  # 使用你的模型
    model.load_state_dict(torch.load(pth_path))

    heatmap = generate_heatmap(model, cropped_image_tensor.squeeze(), target_class)
    overlayed_image = overlay_heatmap(heatmap, cropped_image_tensor.squeeze())

    cv2.imwrite(output_image_path, overlayed_image)
    print(f"Overlayed heatmap saved to: {output_image_path}")

if __name__ == "__main__":
    image_id = '460'
    result_path = 'result/result_ores' 
    main(image_id, result_path)
