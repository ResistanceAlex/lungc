import os

# 数据集文件夹路径
label_folder = './dataset/lungData/labels'  # 替换为label文件夹的路径
images_folder = './dataset/lungData/images'  # 替换为images文件夹的路径

def clean_labels_and_images():
    # 遍历标签文件夹中的所有txt文件
    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_folder, label_file)
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_folder, image_file)

            # 读取标签文件内容
            with open(label_path, 'r') as file:
                lines = file.readlines()

            # 过滤掉类别为0的行
            new_lines = [line for line in lines if not line.startswith('0')]

            # 检查过滤后的内容
            if not new_lines or all(line.strip() == '' for line in new_lines):
                # 如果文件为空或只有换行符，删除对应的图片和标签文件
                os.remove(label_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
                print(f"Deleted: {label_file} and {image_file}")
            else:
                # 如果文件不为空，写回新的内容
                with open(label_path, 'w') as file:
                    file.writelines(new_lines)
                print(f"Updated: {label_file}")

# 执行函数
clean_labels_and_images()
