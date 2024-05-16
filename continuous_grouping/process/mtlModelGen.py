from continuous_grouping.model.mtl_model import MTL_model


def get_models(grouping:list, backbone_name:str, out_features:list):
    models = []
    group_num = max(grouping)
    for i in range(group_num):
        member = [j for j in grouping if grouping[j] == i]
        model = MTL_model(backbone_name=backbone_name, member=member, out_features=out_features)
        models.append(model)
    return models

