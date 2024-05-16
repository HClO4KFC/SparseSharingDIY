import torch


def use_which_loss_func(dataset_name:str, task_name:str):
    # in cvTasks, decide what loss function to use when training separately
    return torch.nn.L1Loss, 'L1Loss'


def use_which_head(dataset:str, task_name:str, in_features:int, out_features:int):
    #
    return torch.nn.Linear(in_features=in_features, out_features=out_features), 'SingleLinear'


class MtlTask():
    # static data structure to store basic information about mtl tasks
    def __init__(self, name:str, no:int, dataset=None):
        self.name = name
        self.no = no
        self.data_set = dataset
        self.loss, self.loss_name = use_which_loss_func(task_name=name)

    def load_dataset(self, dataset):
        self.data_set = dataset


def init_tasks_info(dataset:str):
    task_list = []
    if dataset == 'nyu_v2':
        task_list.append(MtlTask('', 0))
    return task_list
