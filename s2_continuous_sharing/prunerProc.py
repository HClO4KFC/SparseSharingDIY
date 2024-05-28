import copy
import time

import torch


def pruner(args):
    dict_lock = args['dict_lock']
    shared_dict=args['shared_dict']
    task_ranks = args['task_ranks']
    main_tasks = args['main_tasks'] #
    grouping = args['grouping'] #
    task_info_list = args['task_info_list'] #
    single_prune_args = args['single_prune_args']
    batch_per_iter = single_prune_args.batch_per_iter
    dec_percent = single_prune_args.dec_percent
    least_percent = single_prune_args.least_percent
    # 在进行了一段时间的多任务学习后,模型拟合性更好,我们认为此时重新剪枝子网络得到的效果要优于随机初始化下的剪枝结果

    # 对非主要任务按TR值排序
    indexed_arr = list(enumerate(task_ranks))
    # sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1])  # 升序
    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)  # 降序
    sorted_tasks = [index for index, value in sorted_indexed_arr]
    sorted_tasks = [index for index in sorted_tasks if index not in main_tasks]

    # 确定剪枝顺序
    # prune_list = sorted_tasks + main_tasks  # 最后剪枝主要任务, 搭配升序
    prune_list = main_tasks + sorted_tasks  # 最先剪枝主要任务, 搭配降序

    for task_id in prune_list:
        print(f'开始对任务{task_id}:{task_info_list[task_id].name}进行剪枝')

        # 申请所在模型的快照
        dict_lock.acquire_write()
        shared_dict['model' + str(grouping[task_id])] = 'acquired'
        dict_lock.release_write()

        # 尝试获取模型快照
        while True:
            dict_lock.acquire_read()
            if shared_dict['model' + str(grouping[task_id])] != 'acquired':
                model = copy.deepcopy(shared_dict['model' + str(grouping[task_id])])
                break
            dict_lock.release_read()
            time.sleep(1)
        print(f'对任务{task_id}:{task_info_list[task_id].name}的剪枝取得了模型快照')

        # 在快照上进行试训练,制作遮罩
        model.train()
        assert task_id in model.member
        ingroup_no = model.member.index(task_id)
        original_backbone = model.backbone.state_dict()
        original_heads = []
        for head in model.heads:
            original_heads.append(head.state_dict())
        proportion = 1
        optimizer = model.optims[ingroup_no]
        criterion = task_info_list[task_id].loss

        # 初始化遮罩为全通, 子模型参数占比为100%
        for name, param in model.masks[ingroup_no].items():
            with torch.no_grad():
                param.data.fill_(True)

        for i in range(single_prune_args.max_iter):
            # 在单轮i循环里,训练batch_num(k)批数据,然后按比例筛除一部分最小的参数,判断参数占比是否达标,达标则提前跳出
            for batch_idx, (input, target) in enumerate(task_info_list[i].train_set):
                if batch_idx > batch_per_iter:
                    break
                # 在单轮j循环里,进行一批训练,仅用子网参与前向传播,也仅更新子网的参数
                # # 将不在子网中的参数置为0
                # with torch.no_grad():
                #     for name, param in model.backbone.get_named_parameters():
                #         if name in model.pruned_names:
                #             mask = model.masks[ingroup_no][name]
                #             param *= mask.float()
                optimizer.zero_grad()
                output_list = model(input, task_id)  # 子网前向传播
                loss = criterion(output_list[ingroup_no], target)
                loss.backward()
                # 将不在子网中的参数梯度归零, 仅更新子网参数
                with torch.no_grad():
                    for name, param in model.backbone.get_named_parameters():
                        if name in model.pruned_names:
                            mask = model.masks[ingroup_no][name]
                            param.grad *= mask.float()
                optimizer.step()
            # 去掉绝对值最小的<dec_percent>%参数
            for name, param in model.get_named_parameters():
                if name in model.pruned_names:
                    all_items = param.data.abs().flatten()
                    threshold = torch.quantile(all_items, dec_percent / 100)
                    mask = model.masks[ingroup_no][name]
                    mask[param.data.abs() < threshold] = 0
            proportion *= dec_percent / 100
            main_task_tr = [task_ranks[main_id] for main_id in main_tasks if grouping[main_id] == grouping[task_id]][0]
            if proportion <= max(least_percent / 100.0, task_ranks[task_id] / main_task_tr):
                dict_lock.acquire_write()
                shared_dict['mask'+str(task_id)] = model.masks[ingroup_no]
                dict_lock.release_write()
                print(f'任务{task_id}:{task_info_list[task_id].name}已剪枝完成并回传掩膜, 剩余参数{max(least_percent, task_ranks[task_id] / main_task_tr * 100.0)}%')
                break
