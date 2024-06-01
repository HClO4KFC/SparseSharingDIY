import torch.nn
import torchvision

from dlfip.pytorch_object_detection.faster_rcnn.backbone.mobilenetv2_model import MobileNetV2
from dlfip.pytorch_object_detection.faster_rcnn.network_files import FasterRCNN
from dlfip.pytorch_object_detection.faster_rcnn.network_files.rpn_function import AnchorsGenerator, RPNHead


def create_mtl_model(num_classes):
    backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.pth").features
    backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels

    models = []

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率
    models.append(FasterRCNN(backbone=backbone,
                             num_classes=num_classes,
                             rpn_anchor_generator=anchor_generator,
                             box_roi_pool=roi_pooler))

