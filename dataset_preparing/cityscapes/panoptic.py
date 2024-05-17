import os
import cv2
import numpy as np
import json


def load_panoptic_data(image_dir, instance_dir, semantic_dir, annotation_file):
    # 读取图像列表
    image_filenames = sorted(os.listdir(image_dir))
    num_images = len(image_filenames)

    # 加载注释文件
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # 初始化存储数据的列表
    panoptic_data = []

    # 循环处理每张图像
    for image_filename in image_filenames:
        image_id = int(image_filename.split('.')[0])  # 图像ID

        # 读取图像
        image_path = os.path.join(image_dir, image_filename)
        image = cv2.imread(image_path)

        # 读取实例分割掩码
        instance_path = os.path.join(instance_dir, image_filename)
        instance_mask = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)

        # 读取语义分割标签
        semantic_path = os.path.join(semantic_dir, image_filename)
        semantic_mask = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)

        # 从注释中查找当前图像的信息
        annotation = [anno for anno in annotations['annotations'] if anno['image_id'] == image_id][0]
        # 获取实例和类别信息
        instances = annotation['segments_info']

        # 存储图像和对应的实例和语义分割信息
        panoptic_data.append({
            'image': image,
            'instance_mask': instance_mask,
            'semantic_mask': semantic_mask,
            'instances': instances
        })

    return panoptic_data

if __name__ == '__main__':
    # 指定数据集目录和注释文件路径
    image_dir = 'path/to/images'
    instance_dir = 'path/to/instance/masks'
    semantic_dir = 'path/to/semantic/masks'
    annotation_file = 'path/to/annotations.json'

    # 加载数据集
    data = load_panoptic_data(image_dir, instance_dir, semantic_dir, annotation_file)