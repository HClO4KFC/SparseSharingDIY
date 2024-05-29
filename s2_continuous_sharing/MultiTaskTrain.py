import multiprocessing

import torch
from torch.utils.data import DataLoader, Dataset

from utils.lookUpTables import select_cv_task


def multi_task_train(
        dict_lock:multiprocessing.Lock, shared_dict,
        multi_train_args, cv_tasks_args,
        models:list, multi_train_dataset:Dataset,
        task_info_list:list, task_ranks:list,
        grouping:list):

    mtl_data_loader = DataLoader(
        dataset=multi_train_dataset
    )
    for iter_no in range(multi_train_args.max_iter):
        one_ingroup_no_per_group = [select_cv_task(member=model.member, tactic='task_rank', args={'probability':[task_ranks[i] for i in model.member]}) for model in models]
        one_cv_task_per_group = [task_info_list[ingroup_no] for ingroup_no in one_ingroup_no_per_group]
        optims = [models[i].optim[one_ingroup_no_per_group[i]] for i in range(len(models))]
        for batch_idx, (subsets, subset_names) in enumerate(mtl_data_loader):
            inputs = [[subsets[i] for i in range(len(subset_names)) if subset_names[i][0] == cv_tasks_args[ingroup_no]['input']][0] for ingroup_no in one_ingroup_no_per_group]
            targets = [[subsets[i] for i in range(len(subset_names)) if subset_names[i][0] == cv_tasks_args[ingroup_no]['output']][0] for ingroup_no in one_ingroup_no_per_group]
            criterions = [cv_task.loss for cv_task in one_cv_task_per_group]
            for optim in optims:
                optim.zero_grad()
            outputs = [models[group](inputs[group], one_cv_task_per_group[group]) for group in range(len(models))]  # 子网前向传播
            losses = [criterions[group](outputs[group], targets[group]) for group in range(len(models))]
            for loss in losses:
                loss.backward()
            # 将不在子网中的参数梯度滤除, 仅更新子网参数
            with torch.no_grad():
                for group in range(len(models)):
                    for name, param in models[group].backbone.get_named_parameters():
                        if name in models[group].pruned_names:
                            mask = models[group].masks[one_ingroup_no_per_group[group]][name]
                            param.grad *= mask.float()
            for optim in optims:
                optim.step()
            if iter_no > (multi_train_args.warmup_percent / 100.0) * multi_train_args.max_iter:
                for i in range(len(models)):
                    dict_lock.acquire_read()
                    if shared_dict['model' + str(i)] == 'acquired':
                        dict_lock.acquire_write()
                        shared_dict['model' + str(i)] = models[i]
                        dict_lock.release_write()
                    if shared_dict['mask' + str(i)] is not None:
                        model = models[grouping[i]]
                        new_mask = shared_dict['mask' + str(i)]
                        with torch.no_grad():
                            model.masks[model.member.index(i)].copy_(new_mask)
                    dict_lock.release_read()