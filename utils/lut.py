import torch
from init_grouping.data_parsing.cvTask import CvTask


def use_which_loss_func(dataset_name:str, task_name:str):
    # in cvTasks, decide what loss function to use when training separately
    return torch.nn.L1Loss, 'L1Loss'


def use_which_head(dataset:str, task_name:str, in_features:int, out_features:int):
    # in build_head(), calculate the need head structure of model
    return torch.nn.Linear(in_features=in_features, out_features=out_features), 'SingleLinear'


def get_max_patience(train_task_type:str):
    # in TrainTask.tolerate(), get the max time of tolerance before the task got upgraded
    return 1


def init_tasks_info(dataset:str):
    task_list = []
    if dataset == 'nyu_v2':
        task_list.append(CvTask('', 0))
    return task_list
