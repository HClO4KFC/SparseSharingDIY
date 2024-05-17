import random

import torch
from init_grouping.data_parsing.cvTask import CvTask
from utils.errReport import CustomError


def use_which_loss_func(dataset_name: str, task_name: str):
    # in cvTasks, decide what loss function to use when training separately
    return torch.nn.L1Loss, 'L1Loss'


def use_which_optimizer(task_id: int, args: dict):
    # in training process, decide which optimizer to use
    params = args['params']
    lr = args['lr']
    return torch.optim.Adam(params=params, lr=lr)


def use_which_head(dataset: str, task_name: str, in_features: int, out_features: int):
    # in build_head(), calculate the need head structure of model
    return torch.nn.Linear(in_features=in_features, out_features=out_features), 'SingleLinear'


def get_max_patience(train_task_type: str):
    # in TrainTask.tolerate(), get the max time of tolerance before the task got upgraded
    return 1


def init_tasks_info(dataset: str):
    task_list = []
    if dataset == 'nyu_v2':
        task_list.append(CvTask('', 0))
    return task_list


def get_batch_size(cv_task, train_type):
    # in training process, decide the size of one batch
    return 1


def select_cv_task(tactic: str, member: list, args: dict):
    if tactic == 'simple':
        return random.choice(member)
    elif tactic == 'dataset_size' or tactic == 'page_rank':
        prob = args['probability']
        assert len(prob) == len(member)
        prob = prob / sum(prob)
        return random.choices(member, prob)
    else:
        raise CustomError("tactic " + tactic + " is not implemented")


def get_init_lr(task_id):
    return 0.1
