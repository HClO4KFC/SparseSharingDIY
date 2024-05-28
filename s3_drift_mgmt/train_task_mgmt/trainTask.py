import torch.nn

from utils.lut import get_max_patience, get_batch_size, select_cv_task
from utils.errReport import CustomError


class TrainTask:
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
                lr = self.args['lr'],
                remain_percent=self.args['prune_remain_percent'],
                decrease_percent=self.args['dec_percent'])
        else:
            raise CustomError("train_type "+self.train_type+" is not defined")

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

    def solve_as_prune(self, lr:float, remain_percent:float, decrease_percent:float, batch_num:int, batch_size:int, prune_lr:float):
        # 在进行了一段时间的多任务学习后,模型拟合性更好,我们认为此时重新剪枝子网络得到的效果要优于随机初始化下的剪枝结果
        task_id = self.cv_tasks[0].no
        model = self.model
        model.train()
        assert task_id in model.member
        ingroup_no = model.member.index(task_id)
        original_backbone = model.backbone.state_dict()
        original_heads = []
        for head in model.heads:
            original_heads.append(head.state_dict())
        percentile = 1
        optimizer = model.optims[ingroup_no]
        criterion = self.cv_tasks[0].loss
        # 初始化遮罩为全通, 子模型参数占比为100%
        for name, param in model.masks[ingroup_no].items():
            with torch.no_grad():
                param.data.fill_(True)

        for i in range(self.max_iter):
            # 在单轮i循环里,训练batch_num(k)批数据,然后按比例筛除一部分最小的参数,判断参数占比是否达标,达标则提前跳出
            for j in range(batch_num):
                # 在单轮j循环里,进行一批训练,仅用子网参与前向传播,也仅更新子网的参数
                batch_x, batch_y = self.cv_tasks[0].get_next_batch(batch_size)
                # # 将不在子网中的参数置为0
                # with torch.no_grad():
                #     for name, param in model.backbone.get_named_parameters():
                #         if name in model.pruned_names:
                #             mask = model.masks[ingroup_no][name]
                #             param *= mask.float()
                optimizer.zero_grad()
                output_list = model(batch_x, task_id)  # 子网前向传播
                loss = criterion(output_list[ingroup_no], batch_y)
                loss.backward()
                # 将不在子网中的参数梯度归零, 仅更新子网参数
                with torch.no_grad():
                    for name, param in model.backbone.get_named_parameters():
                        if name in model.pruned_names:
                            mask = model.masks[ingroup_no][name]
                            param.grad *= mask.float()
                optimizer.step()
            # 去掉绝对值最小的<decrease_percent>%参数
            for name, param in model.get_named_parameters():
                if name in model.pruned_names:
                    all_items = param.data.abs().flatten()
                    threshold = torch.quantile(all_items, decrease_percent / 100)
                    mask = model.masks[ingroup_no][name]
                    mask[param.data.abs() < threshold] = 0
            percentile *= decrease_percent / 100
            if percentile <= remain_percent:
                break
        model.backbone.load_state_dict(original_backbone)
        for i in range(len(original_heads)):
            model.heads[i].load_state_dict(original_heads[i])

    def solve_as_mtl_train(self, max_mtl_iter:int, train_type:str):
        model=self.model
        for i in range(max_mtl_iter):
            cv_task = select_cv_task(member=model.member, tactic='simple', args={})
            ingroup_no = model.member.index()
            optimizer = model.optim[ingroup_no]
            batch_x, batch_y = cv_task.get_next_batch(get_batch_size(cv_task=cv_task, train_type=train_type))
            criterion = cv_task.loss
            optimizer.zero_grad()
            output = model(batch_x, cv_task)  # 子网前向传播
            loss = criterion(output, batch_y)
            loss.backward()
            # 将不在子网中的参数梯度滤除, 仅更新子网参数
            with torch.no_grad():
                for name, param in model.backbone.get_named_parameters():
                    if name in model.pruned_names:
                        mask = model.masks[ingroup_no][name]
                        param.grad *= mask.float()
            optimizer.step()
