import torch
from torch import nn
from torchvision.ops import FrozenBatchNorm2d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, base_width=64,
                 groups=1, stride=1, downsample=None, norm_layer=None, dilation=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 模型内部的卷积和bn层不采用out_channel作为宽度,而使用out_channel(另:planes), base_width和groups算出的width(模型内部宽度),原理和原因待考证
        width = int(out_channel * (base_width / 64.)) * groups  # WTF is the 64.(from fcn, backbone.py)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = norm_layer(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width,
                               kernel_size=3, stride=stride, bias=False,
                               padding=dilation, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        # -----------------------------------------
        # 此处没有用width而使用了planes, 原因可能是已经不属于block"内部"了,使用其与外部链接的接口宽度planes
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000,
                 norm_layer=None,
                 replace_stride_with_dilation=None,
                 zero_init_residual=None,
                 groups=1, width_per_group=64):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channel = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # fcn resnet definition:
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, channel, block_num, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,
                                norm_layer=norm_layer, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # if self.include_top:->
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps
