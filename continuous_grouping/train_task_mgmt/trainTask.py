import torch.nn

from continuous_grouping.process.lut import get_max_patience

class TrainTask():
    # 用于描述训练任务的类,在trainQueue中管理,注意不是多任务学习中的视觉任务(本代码中一般称为cv_task)

    def __init__(self, model:torch.nn.Module, cv_tasks:list, max_epoch:int, train_type:str, args:dict):
        # belonging
        self.model = model
        self.cv_tasks = cv_tasks.sort()
        self.max_epoch = max_epoch
        self.train_type = train_type
        self.belongings = args
        # stage
        self.patience = 0
        self.queue_lvl = -1
        self.stage = 'waiting'

    def solve(self):
        # TODO:执行当前训练任务
        pass

    def grade(self):
        # TODO:评估当前任务在同等优先级下的收敛速度,用于任务调度
        pass

    def is_same_as(self, train_task:'TrainTask'):
        if self.model != train_task.model or self.max_epoch != train_task.max_epoch or self.train_type != train_task.train_type:
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
