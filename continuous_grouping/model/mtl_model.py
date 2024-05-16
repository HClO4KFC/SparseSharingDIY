import torch.nn

from utils.errReport import CustomError
from continuous_grouping.model.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
from continuous_grouping.process.lut import use_which_head


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
    def __init__(self, backbone_name:str, member:list, out_features:list):
        super(MTL_model, self).__init__()
        self.backbone = build_backbone(backbone_name)
        self.heads = []
        self.head_names = []
        self.out_features = out_features
        for i in member:
            head, head_name = build_head(task_id=i, in_size=self.backbone.out_channels, out_size=out_features[i])
            self.heads.append(head)
            self.head_names.append(head_name)

    def forward(self, x):
        out = self.backbone(x)
        out_list = []
        for head in self.heads:
            out_list.append(head(out))
        return out_list
