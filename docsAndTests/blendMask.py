import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            l_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            o_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.output_convs.append(o_conv)

    def forward(self, x):
        last_inner = self.lateral_convs[-1](x[-1])
        results = [self.output_convs[-1](last_inner)]
        for feature, l_conv, o_conv in zip(
            x[:-1][::-1], self.lateral_convs[:-1][::-1], self.output_convs[:-1][::-1]
        ):
            lat = l_conv(feature)
            last_inner = F.interpolate(last_inner, size=lat.shape[-2:], mode="nearest") + lat
            results.insert(0, o_conv(last_inner))
        return results

class BlendModule(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(BlendModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, global_feat, local_feat):
        x = torch.cat([global_feat, local_feat], dim=1)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class BlendMask(nn.Module):
    def __init__(self, num_classes):
        super(BlendMask, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.fpn = FPN([256, 512, 1024, 2048], 256)
        self.blend_module = BlendModule(256 + 256, 128, num_classes)
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.backbone(x)
        features = [features[layer] for layer in [2, 3, 4, 5]]  # Extracting C2, C3, C4, C5 layers
        features = self.fpn(features)

        global_feats = features[0]
        cls_preds = self.cls_head(global_feats)

        mask_feats = self.mask_head(global_feats)
        seg_preds = self.blend_module(global_feats, mask_feats)

        return cls_preds, seg_preds

# Example usage
model = BlendMask(num_classes=80)
input_tensor = torch.randn(1, 3, 512, 512)
cls_preds, seg_preds = model(input_tensor)