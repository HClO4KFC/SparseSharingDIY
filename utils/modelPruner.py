import torch


class Pruner:
    def __init__(self, model:torch.nn.Module,
                 prune_names:list,
                 remain_percent:float,
                 max_pruning_iter:int):
        # TODO
        pass
