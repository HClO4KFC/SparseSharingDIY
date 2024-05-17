import copy

import torch.nn

from utils.errReport import CustomError
from init_grouping.model.mtl_net.resNet import resnet18, resnet34, resnet50, resnet101, resnet152
from utils.lut import use_which_head, use_which_optimizer, get_init_lr


def build_backbone(backbone_name):
    if backbone_name == 'ResNet18':
        model = resnet18()
    elif backbone_name == 'ResNet34':
        model = resnet34()
    elif backbone_name == 'ResNet50':
        model = resnet50()
    elif backbone_name == 'ResNet101':
        model = resnet101()
    elif backbone_name == 'ResNet152':
        model = resnet152()
    else:
        raise CustomError("backbone " + backbone_name + " is not implemented yet")
    return model


def build_head(task_id, in_size, out_size):
    # TODO: try use meta learning or NAS method to decide the structure of head of each task
    model, model_name = use_which_head(task_id, in_features=in_size, out_features=out_size)
    return model, model_name


class MTL_model(torch.nn.Module):
    def __init__(self, backbone_name:str, member:list, out_features:list, prune_names:list):
        super(MTL_model, self).__init__()
        self.backbone = build_backbone(backbone_name)
        self.heads = []
        self.head_names = []
        self.out_features = out_features
        self.member = member
        self.pruned_names = prune_names
        self.masks = []
        self.optims = []
        for i in member:
            head, head_name = build_head(task_id=i, in_size=self.backbone.out_channels, out_size=out_features[i])
            self.heads.append(head)
            self.head_names.append(head_name)
            mask = {
                name: torch.nn.Parameter(torch.ones(named_param.size()).to(named_param).bool(), requires_grad=False)
                for name, named_param in self.backbone.named_parameters()
                if name in self.pruned_names}
            self.masks.append(mask)
        self.randomly_initialize_weights()
        for i in member:
            self.optims.append(use_which_optimizer(task_id=i, args={'params': self.get_parameters(), 'lr': get_init_lr(i)}))

    def randomly_initialize_weights(self):
        # Initialize backbone parameters
        for name, param in self.backbone.named_parameters():
            param.data = torch.rand(param.size())

        # Initialize heads parameters
        for head in self.heads:
            for name, param in head.named_parameters():
                param.data = torch.rand(param.size())

    def forward(self, x, task_id:int):
        assert task_id in self.member
        ingroup_no = self.member.index(task_id)
        masked_params = {}
        # 构造子网络参数masked_param,
        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name in self.pruned_names:
                    mask = self.masks[ingroup_no][name]
                    masked_param = param * mask.float()
                    masked_params[name] = masked_param

        # 保存并替换原参数为子层参数, 用子层参数做前向传播
        original_params = {}
        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name in masked_params:
                    original_params[name] = param.data.clone()
                    param.data.copy_(masked_params[name])

        # 前向传播
        out = self.backbone(x)

        # 恢复原始参数
        with torch.no_grad():
            for name, param in self.backbone.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

        out_list = []
        for head in self.heads:
            out_list.append(head(out))
        return out_list


class SubModel():
    def __init__(self, model:torch.nn.Module, ingroup_no:int):
        self.backbone = copy.deepcopy(model.backbone)
        self.head = copy.deepcopy(model.heads[ingroup_no])
        self.mask = copy.deepcopy(model.masks[ingroup_no])
        self.pruned_names = copy.deepcopy(model.pruned_names)

        with torch.no_grad():
            for name, param in self.backbone.get_named_parameters():
                if name in self.pruned_names:
                    param_mask = self.mask[ingroup_no][name]
                    param *= param_mask.float()

    def forward(self, x):
        # 前向传播
        out = self.backbone(x)
        out = self.head(out)
        return out
