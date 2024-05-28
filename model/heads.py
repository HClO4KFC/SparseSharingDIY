import torch.nn
from torchvision.models.detection import MaskRCNN
import torch
import torch.nn as nn
import torchvision.ops as ops


class GenDisparityHead(torch.nn.Module):
    def __init__(self, cv_task_arg, in_features, out_features=1):
        super(GenDisparityHead, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_features, out_channels=256, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=cv_task_arg.scale_factor, mode=cv_task_arg.upsample_mode),
            torch.nn.Conv2d(in_channels=256, out_channels=out_features, kernel_size=1)  #输出深度图的通道数为1
        )

    def forward(self, x):
        depth_map = self.decoder(x)
        return depth_map


class GenRightHead(torch.nn.Module):
    def __init__(self, cv_task_arg, in_features, out_features):
        super(GenRightHead, self).__init__()
        self.decoder = torch.nn.Sequential(
            # TODO
        )


class ObjDetHead(torch.nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(ObjDetHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # 回归头
        self.reg_head = torch.nn.Conv2d(256, num_anchors * 4, kernel_size=1)

        # 分类头
        self.cls_head = torch.nn.Conv2d(256, num_anchors * num_classes, kernel_size=1)

    def forward(self, x):
        reg_out = self.reg_head(x)
        cls_out = self.cls_head(x)

        # 调整形状
        N, _, H, W = x.shape
        reg_out = reg_out.view(N, self.num_anchors, 4, H, W).permute(0, 2, 1, 3, 4).contiguous()
        cls_out = cls_out.view(N, self.num_anchors, self.num_classes, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # 回归输出形状：(N, 4,           num_anchors, H, W)
        # 分类输出形状：(N, num_classes, num_anchors, H, W)
        return torch.cat((reg_out, cls_out), dim=1)


class LabelSegHead(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LabelSegHead, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        self.upsample = torch.nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

    ## 示例用法
    # model = ResNet50Segmentation(num_classes=21)  # 假设有21个类别（包括背景）
    # input_tensor = torch.randn(8, 3, 224, 224)    # 假设输入张量的形状是 (batch_size=8, channels=3, height=224, width=224)
    # output_tensor = model(input_tensor)
    # print(output_tensor.shape)  # 输出张量的形状应该是 (8, 21, 224, 224)


class InstSegHead(torch.nn.Module):
    def __init__(self, num_classes):
    # FPN
        self.fpn = ops.FeaturePyramidNetwork([256, 512, 1024, 2048], 256)

        # ROIAlign
        self.roi_align = ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                output_size=7,
                                                sampling_ratio=2)

        # Mask Head
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, images, proposals):
        # 主干网络
        features = self.backbone(images)

        # FPN
        fpn_features = self.fpn(features)

        # ROIAlign
        roi_features = self.roi_align(fpn_features, proposals, image_shapes=[images.shape[-2:]] * len(proposals))

        # Mask Head
        mask_logits = self.mask_head(roi_features)

        return mask_logits

    # 示例用法
    # num_classes = 2  # 假设有两个类别
    # model = MaskRCNN(num_classes=num_classes)
    # images = torch.randn(1, 3, 224, 224)  # 示例输入
    # proposals = [torch.tensor([[50, 50, 100, 100], [30, 30, 70, 70]], dtype=torch.float32)]  # 示例提案
    # output = model(images, proposals)
    # print(output.shape)  # 输出形状为 [batch_size, num_rois, num_classes, mask_size, mask_size]


class PanopticHead(torch.nn.Module):
    def __init__(self):
        #TODO
        pass

    def forward(self, x):
        #TODO
        pass
