import random
import torch
from torchvision.transforms import transforms

from datasets.dataLoader import SingleDataset
from utils.errReport import CustomError


def use_which_loss_func(task_name: str):
    # in cvTasks, decide what loss function to use when training separately
    print('实现lut.use_which_loss_func')
    if task_name == 'genDisparity':
        return torch.nn.SmoothL1Loss(), 'SmoothL1Loss'
    else:
        return torch.nn.L1Loss, 'L1Loss'


def use_which_optimizer(task_id: int, args: dict):
    # in training process, decide which optimizer to use
    params = args['params']
    lr = args['lr']
    return torch.optim.Adam(params=params, lr=lr)


def use_which_head(task_name: str, in_features: int, out_features: int):
    # in build_head(), calculate the need head structure of model
    print('实现use_which_head')
    return torch.nn.Linear(in_features=in_features, out_features=out_features), 'SingleLinear'


def get_max_patience(train_task_type: str):
    # in TrainTask.tolerate(), get the max time of tolerance before the task got upgraded
    print('这里用了写死的patience=1')
    return 1


def get_batch_size(cv_task, train_type):
    # in training process, decide the size of one batch
    print('这里用了写死的batch size = 8')
    return 8


def select_cv_task(tactic: str, member: list, args: dict):
    ingroup_no = [i for i in range(len(member))]
    if tactic == 'simple':
        chosen = random.choice(ingroup_no)
    elif tactic == 'dataset_size' or tactic == 'task_rank':
        prob = args['probability']
        assert len(prob) == len(member)
        prob = prob / sum(prob)
        chosen = random.choices(ingroup_no, prob)
    else:
        raise CustomError("tactic " + tactic + " is not implemented")
    return chosen


def get_init_lr(task_id):
    # task_id = -1指的是backbone
    print('这里用了写死的init lr = 0.1')
    return 0.1


def collate_func(sub_out, subset_name):
    print('实现collate_func')
    return sub_out, subset_name


class CvTask:
    # static data structure to store basic information about mtl tasks
    def __init__(self, no: int, dataset_args,
                 cv_task_arg, cv_subsets_args):
        self.name = cv_task_arg.name
        self.no = no
        self.loss, self.loss_name = use_which_loss_func(task_name=self.name)
        self.train_set = SingleDataset(dataset=dataset_args.dataset_name,
                                       path_pre=dataset_args.path_pre,
                                       cv_task_arg=cv_task_arg,
                                       cv_subsets_args=cv_subsets_args,
                                       train_val_test='train',
                                       transform=transforms.Compose([transforms.ToTensor()]))
        self.val_set = SingleDataset(dataset=dataset_args.dataset_name,
                                     path_pre=dataset_args.path_pre,
                                     cv_task_arg=cv_task_arg,
                                     cv_subsets_args=cv_subsets_args,
                                     train_val_test='val',
                                     transform=transforms.Compose([transforms.ToTensor()]))
        # self.train_loader = DataLoader(dataset=self.train_set,
        #                                batch_size=dataset_args.batch_size,
        #                                shuffle=True,
        #                                sampler=None,
        #                                collate_fn=collate_func)
        # self.val_loader = DataLoader(dataset=self.val_set,
        #                              batch_size=dataset_args.batch_size,
        #                              shuffle=False,
        #                              sampler=None,
        #                              collate_fn=collate_func)

    # def get_next_batch(self, train_val_test: str):
    #     if train_val_test == 'train':
    #         return self.train_loader
    #     print('实现CvTask.get_next_batch()')
    #     # TODO: 从dataset中取出从dataset_bookmark开始,长度为batch_size的一批数据,并更新dataset_bookmark
    #     pass
