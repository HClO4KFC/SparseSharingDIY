import copy

import torch.nn

from utils.lut import get_max_patience
from utils.errReport import CustomError


class TrainTask():
    # 用于描述训练任务的类,在trainQueue中管理,注意不是多任务学习中的视觉任务(本代码中一般称为cv_task)

    def __init__(self, model:torch.nn.Module, cv_tasks:list, max_epoch:int, train_type:str, args:dict):
        # belonging
        self.model = model
        self.cv_tasks = cv_tasks.sort()
        self.max_iter = max_epoch
        self.train_type = train_type
        self.args = args
        # stage
        self.patience = 0
        self.queue_lvl = -1
        self.stage = 'waiting'

    def solve(self):
        if self.train_type == 'idle':
            return
        elif self.train_type == 'multi_task_train':
            # 对多任务模型进行所有任务上的学习
            self.solve_as_mtl_train()
        elif self.train_type == 'multi_task_warmup':
            self.solve_as_mtl_train()
        elif self.train_type == 'single_task_prune':
            # 学习多任务模型中的单任务子模型结构
            self.solve_as_prune(
                remain_percent=self.args['prune_remain_percent'],
                decrease_percent=self.args['decrease_percent'])
        else:
            raise CustomError("train_type "+self.train_type+" is not defined")
        pass

    def grade(self):
        # TODO:评估当前任务收敛速度,用于在同等优先级下的任务调度
        pass

    def is_same_as(self, train_task:'TrainTask'):
        if self.model != train_task.model or self.max_iter != train_task.max_iter or self.train_type != train_task.train_type:
            return False
        if len(self.cv_tasks) != len(train_task.cv_tasks):
            return False
        for i in range(len(self.cv_tasks)):
            if self.cv_tasks[i].no != train_task.cv_tasks[i].no:
                return False
        return True

    def tolerate(self):
        self.patience += 1
        # 若没到忍耐限度,就忍(返回True),否则返回False,会被调度到高优先级队列
        if self.patience < get_max_patience(self.train_type):
            return 1
        else:
            self.patience = 0
            return 0

    def get_result(self):
        # TODO: trainer训练完成后发给manager,进而发给main的重训练结果
        pass

    def solve_as_prune(self, remain_percent:float, decrease_percent:float):
        model = self.model
        model.train()
        original_params = copy.deepcopy(model.get_parameter())
        # set mask to all 1
        for i in range(self.max_iter):
            batch_x, batch_y = self.cv_tasks[0].get_next_batch()
            # train with one batch of data
            # cut off the least significant <decrease_percent>% of params

    def solve_as_mtl_train(self):
        # TODO: multi task training
        pass








