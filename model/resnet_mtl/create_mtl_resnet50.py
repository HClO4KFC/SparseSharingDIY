import torch
from torchvision.ops import FrozenBatchNorm2d

from model.resnet_mtl.backbone_with_fpn import BackboneWithFPN, LastLevelMaxPool, IntermediateLayerGetter
from model.resnet_mtl.resnet_model import ResNet, Bottleneck, overwrite_eps


def mtl_resnet50_backbone(
        replace_stride_with_dilation,
        aux,
        norm_layer=FrozenBatchNorm2d,
        trainable_layers=3,
        returned_layers=None,
        extra_blocks=None):

    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3],
                             norm_layer=norm_layer,
                             replace_stride_with_dilation=replace_stride_with_dilation)


    #
    # if pretrain_path != "":
    #     assert os.path.exists(pretrain_path), "{} is not exist.".format(pretrain_path)
    #     # 载入预训练权重
    #     print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))

    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)
        # 调整批归一化层的epsilon参数,提高模型数值稳定性

    out_inplanes = 2048
    aux_inplanes = 1024

    # return_layers = {'layer4': 'out'}
    # if aux:
    #     return_layers['layer3'] = 'aux'

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果要训练所有层结构的话，不要忘了conv1后还有一个bn1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # freeze layers
    for name, parameter in resnet_backbone.named_parameters():
        # 只训练不在layers_to_train列表中的层结构
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    # seg任务中,默认所有层都要
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    # 返回的特征层个数肯定大于0小于5
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    # return_layers['layer4'] = 'out'
    # if aux:
    #     return_layers['layer3'] = 'aux'

    # resnet_backbone = IntermediateLayerGetter(resnet_backbone, return_layers)

    # in_channel 为layer4的输出特征矩阵channel = 2048
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    # 记录resnet50提供给fpn的每个特征层channel
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # 通过fpn后得到的每个特征层的channel
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
