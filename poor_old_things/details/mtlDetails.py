from model.mtlModel import ModelTree


def get_models(grouping:list, backbone_name:str, out_features:list, prune_names:list):
    models = []
    group_num = max(grouping)
    for i in range(group_num):
        member = [j for j in grouping if grouping[j] == i]
        model = ModelTree(backbone_name=backbone_name, member=member, out_features=out_features, prune_names=prune_names)
        models.append(model)
    return models

