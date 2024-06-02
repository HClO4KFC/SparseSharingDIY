import copy

import torch.nn
from torch import nn
from torch.nn import init
from torchvision import models

from dlfip.pytorch_object_detection.faster_rcnn.backbone import MobileNetV2
from dlfip.pytorch_object_detection.faster_rcnn.network_files import FasterRCNN, FastRCNNPredictor
from dlfip.pytorch_segmentation.fcn.src.fcn_model import FCNHead, FCN
from model.resnet_mtl.create_mtl_resnet50 import mtl_resnet50_backbone
from utils.errReport import CustomError
from model.resNet import resnet18, resnet34, resnet101, resnet152


# from utils.lookUpTables import use_which_head, use_which_optimizer, get_init_lr


def build_backbone(backbone_name):
    if backbone_name == 'ResNet18':
        model = resnet18()
        out_channels = model.out_channels
    elif backbone_name == 'ResNet34':
        model = resnet34()
        out_channels = model.out_channels
    elif backbone_name == 'ResNet50':
        # model = resnet50()
        model = mtl_resnet50_backbone(
            aux=False,
            norm_layer=torch.nn.BatchNorm2d,
            trainable_layers=5,
            replace_stride_with_dilation=[False, True, True],
            returned_layers=None,
            extra_blocks=None)
        # print(model)
    elif backbone_name == 'ResNet101':
        model = resnet101()
        out_channels = model.out_channels
    elif backbone_name == 'ResNet152':
        model = resnet152()
        out_channels = model.out_channels
    elif backbone_name == 'MobileNetV3Small':
        model = models.mobilenet_v3_small(pre_trained=False)
        model = torch.nn.Sequential(*list(model.children())[:-2])
        out_channels = list(model.children())[0][-1][0].out_channels
    elif backbone_name =='MobileNetV2':
        backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.pth").features
        backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels
    else:
        raise CustomError("backbone " + backbone_name + " is not implemented yet")
    return model


# def build_head(cv_task_arg, in_size, out_size):
#     # TODO: try use meta learning or NAS method to decide the structure of head of each task
#     model, model_name = use_which_head(task_name=cv_task_arg.name, in_features=in_size, out_features=out_size)
#     return model, model_name


class ModelTree(torch.nn.Module):
    def __init__(self, backbone_name: str, member: list, out_features: list, prune_names: list, cv_tasks_args,
                 no_mask=False):
        super(ModelTree, self).__init__()
        # self.no_mask = no_mask
        self.backbone = build_backbone(backbone_name)
        # print(self.backbone)
        self.tasks = []
        # self.head_names = []
        # self.out_features = out_features
        self.member = member
        self.pruned_names = prune_names
        self.masks = []
        self.optims = []

        for i in member:
            if i == 0:
                # fcn
                out_inplanes = 2048
                aux_inplanes = 1024
                aux = False
                aux_classifier = None
                if aux:
                    aux_classifier = FCNHead(
                        in_channels=aux_inplanes,
                        channels=21)
                classifier = FCNHead(
                    in_channels=out_inplanes,
                    channels=21)
                model = FCN(self.backbone, classifier, aux_classifier)
            else:
                #faster-rcnn
                model = FasterRCNN(backbone=self.backbone)
                # get number of input features for the classifier
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                # replace the pre-trained head with a new one
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=6)
                pass
            mask = {name: torch.nn.Parameter(torch.ones(named_param.size()).to(named_param).bool(), requires_grad=False)
                    for name, named_param in self.backbone.named_parameters()
                    if named_param.requires_grad}
                    # if name in prune_names}
            self.masks.append(mask)
            self.tasks.append(model)

        #     # head, head_name = build_head(cv_task_arg=cv_tasks_args[i], in_size=backbone_out_channels,
        #     #                              out_size=out_features[i])
        #     # self.heads.append(head)
        #     # self.head_names.append(head_name)
        #
        # for i in member:
        #     self.optims.append(use_which_optimizer(task_id=i, args={'params': self.parameters(), 'lr': get_init_lr(i)}))

    def randomly_initialize_weights(self):
        # Initialize backbone parameters
        for name, param in self.backbone.named_parameters():
            param.data = torch.rand(param.size())

        # Initialize heads parameters
        for head in self.heads:
            for name, param in head.named_parameters():
                param.data = torch.rand(param.size())

    def forward(self, x, task_id=-1):
        if task_id == -1:  # 整体训练
            if self.no_mask:
                out = self.backbone(x)
                outs = [self.heads[i](out) for i in range(len(self.heads))]
                return outs
            outs = [self.forward(x, i) for i in self.member]
            return outs
        assert task_id in self.member
        ingroup_no = self.member.index(task_id)
        masked_params = {}
        # 构造子网络参数masked_param,
        with torch.no_grad():
            params = self.backbone.named_parameters()
            for name, param in params:
                mask = self.masks[ingroup_no][name]
                masked_param = param * mask.float()
                masked_params[name] = masked_param
                assert masked_params[name].shape == param.shape

        # 保存并替换原参数为子层参数, 用子层参数做前向传播
        original_params = {}
        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name in masked_params:
                    original_params[name] = param.data.clone()
                    param.data.copy_(masked_params[name])
                    assert param.shape == original_params[name].shape

        # 前向传播
        out = self.backbone(x)

        # 恢复原始参数
        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

        # out_list = []
        # for head in self.heads:
        #     out_list.append(head(out))
        # return out_list
        out = self.heads[ingroup_no](out)
        return out

    def get_part(self, allocation: list):
        ans = copy.deepcopy(self)
        set_a = set(ans.member)
        set_b = set(allocation)
        assert set_b.issubset(set_a)
        diff = set_a - set_b
        del_idx_list = [i for i in range(len(self.member)) if i in diff]
        for del_idx in del_idx_list:
            self.heads.pop(del_idx)
            self.head_names.pop(del_idx)
            self.out_features.pop(del_idx)
            self.member.pop(del_idx)
            self.masks.pop(del_idx)
            self.optims.pop(del_idx)
        return ans

    def get_subset_mapping(self, io: str, cv_subset_args, cv_task_args):
        if io == 'input':
            ans = [[subset_no for subset_no in range(len(cv_subset_args)) if
                    cv_subset_args[subset_no].name == cv_task_args[member].input][0] for member in self.member]
        elif io == 'output':
            ans = [[subset_no for subset_no in range(len(cv_subset_args)) if
                    cv_subset_args[subset_no].name == cv_task_args[member].output][0] for member in self.member]
        else:
            assert False
        return ans


class ModelForrest(torch.nn.Module):
    def __init__(self, tree_list: list, allocation: list, cv_subset_args, cv_task_args):
        super(ModelForrest, self).__init__()
        self.models = [tree_list[i].get_part(allocation[i]) for i in range(len(tree_list))]
        self.input_subset_mapping = [self.models[i].get_subset_mapping('input', cv_subset_args, cv_task_args) for i in
                                     range(len(self.models))]
        self.output_subset_mapping = [self.models[i].get_subset_mapping('output', cv_subset_args, cv_task_args) for i in
                                      range(len(self.models))]
        self.task_mapping = [self.models[i].member for i in range(len(self.models))]
        # self.backbone = copy.deepcopy(model.backbone)
        # self.head = copy.deepcopy(model.heads[ingroup_no])
        # self.mask = copy.deepcopy(model.masks[ingroup_no])
        # self.pruned_names = copy.deepcopy(model.pruned_names)

    def forward(self, x):
        # 前向传播
        # out = self.backbone(x)
        # out = self.head(out)
        outs = [model(x) for model in self.models]
        return outs

    def get_tree_list(self):
        return self.models

    def update(self, update_type: str, update_pack: dict):
        if update_type == 'model update':
            model_id = update_pack['model_id']
            new_model = update_pack['new_model']
            assert len(self.models[model_id].member) == len(new_model.member)
            for i in range(len(self.models[model_id].member)):
                assert self.models[model_id].member[i] == new_model.member[i]
            self.models[model_id] = new_model  # [new_model.get_part(allocation[i]) for i in range(len(tree_list))]
        else:
            raise CustomError('unknown update type: ' + update_type)
