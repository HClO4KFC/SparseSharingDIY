import random
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from s1_init_structure.datasets.dataLoader import MultiDataset
from model.mtlModel import MTL_model
from s1_init_structure.mtg_net.transformer import HOINetTransformer
from poor_old_things.trainEvalTest import mtg_training
from utils.lut import get_init_lr


def try_mtl_train(multi_train_dataset: MultiDataset, backbone: str, grouping: list, out_features: list,
                  try_epoch_num: int, tasks_info_list: list, cv_tasks_args) -> list:
    member = [i for i in range(len(grouping)) if grouping[i] == 1]
    model = MTL_model(
        backbone_name=backbone,
        member=member,
        out_features=out_features,
        prune_names=[],
        cv_tasks_args=cv_tasks_args)
    multi_train_dataLoader = DataLoader(
        dataset=multi_train_dataset,
        batch_size=2,
        drop_last=True,
        shuffle=True,
    )
    heads_optims = [Adam(params=model.heads[i].parameters(), lr=get_init_lr(model.member[i])) for i in
                    range(len(model.member))]
    backbone_optim = Adam(params=model.backbone.parameters(), lr=get_init_lr(-1))
    criterions = [tasks_info_list[i].loss for i in member]
    for epoch in range(try_epoch_num):
        model.train()
        for batch_idx, (subsets, subset_names) in enumerate(multi_train_dataLoader):
            # 整理批数据
            data = [subsets[i] for i in range(len(subset_names)) if subset_names[i][0] == cv_tasks_args[0]['input']][0]
            targets = []
            for cv_task_no in member:
                targets.append([subsets[i] for i in range(len(subset_names)) if subset_names[i][0] == cv_tasks_args[cv_task_no]['output']][0])

            for ingroup_no in range(len(member)):
                backbone_optim.zero_grad()
                heads_optims[ingroup_no].zero_grad()
                outputs = model(x=data, task_id=-1)
                loss = criterions[ingroup_no](outputs[ingroup_no], targets[ingroup_no])
                loss.backward()
                heads_optims[ingroup_no].step()
                backbone_optim.step()

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{try_epoch_num}, Task {cv_task_no}, Batch {batch_idx + 1}/{len(multi_train_dataset)}, "
                        f"Loss: {loss.item():.4f}")


def binList2Int(binList: list) -> int:
    ans = 0
    base = 1
    for i in range(len(binList)):
        ans += base * binList[i]
        base *= 2
    return ans


def int2BinList(x: int, lenth: int) -> list:
    ans = []
    while x != 0:
        ans.append(1 if x % 2 == 1 else 0)
        x /= 2
    while len(ans) < lenth:
        ans.append(0)
    return ans


def mtg_active_learning(multi_train_dataset, init_grp_args, mtgnet_args,
                        dataset_name, gpu_id, backbone, out_features,
                        task_info_list, cv_task_args):
    mtgnet_upd_freq=init_grp_args.mtgnet_upd_freq
    high_gain_preference=init_grp_args.high_gain_prefer
    train_set_size=init_grp_args.train_set_develop * init_grp_args.meta_train_iter
    try_epoch_num=init_grp_args.labeling_try_epoch
    new_train_per_iter=init_grp_args.train_set_develop
    num_layers=mtgnet_args.num_layers
    num_hidden=mtgnet_args.num_hidden
    dropout_rate=mtgnet_args.dropout_rate
    device='cuda:' + gpu_id
    ensemble_num=init_grp_args.ensemble_num
    task_num = len(cv_task_args)

    train_x = []
    train_y = []
    ensemble_model = []
    ensemble_model_loss = []
    in_test_list = [True for _ in range(2 ** (task_num + 1))]
    train_index = 0  # 下次从train_x[train_index]开始训练
    model = HOINetTransformer(
        num_layers=num_layers,
        model_dim=num_hidden,
        num_heads=1,
        task_dim=task_num,
        ffn_dim=num_hidden,
        dropout=dropout_rate
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.007,
        weight_decay=0.0005
    )

    # 开始主动学习, 首批训练集仅包括"全分到一组"这一种情况
    selected_train_set = [[1 for _ in range(task_num)]]
    in_test_list[binList2Int([1 for _ in range(task_num)])] = False
    baseline_x = [[1 if i == j else 0 for i in range(task_num)] for j in range(task_num)]  # 此后所有训练集都要对标这些单任务模型
    baseline_y = []
    for x in baseline_x:
        in_test_list[binList2Int(x)] = False
    for x in baseline_x:
        baseline_output = try_mtl_train(multi_train_dataset=multi_train_dataset, grouping=x, try_epoch_num=try_epoch_num,
                                        backbone=backbone, out_features=out_features,
                                        tasks_info_list=task_info_list, cv_tasks_args=cv_task_args)
        assert len([i for i in baseline_output if i != 0]) == 1
        baseline_y.append(sum(baseline_output))
        # try_mtl_train()应接受一个0,1数组,在为1的任务中构建硬参数共享模型,试训练try_epoch_num个周期,记录并返回这段时间组内每个任务loss变化的斜率
        # 此外,为了debug用,应保存试训练期间的loss变化到文件
    for k in range(train_set_size // new_train_per_iter):
        for j in range(task_num):
            # 制作上批选出的ctrain的ground truth, 清空候选
            for new_member in selected_train_set:
                new_member_output = try_mtl_train(grouping=new_member, try_epoch_num=try_epoch_num)
                train_x.append(new_member)
                train_y.append([new_member_output[i] - baseline_y[i] for i in range(task_num) if baseline_y[i] != 0])

            # 选出c_test中所有包含T_j的方案
            candidates = []
            for i in range(2 ** task_num):
                no = int2BinList(i, task_num - 1)
                no.insert(j, 1)
                if in_test_list[binList2Int(no)]:
                    candidates.append(no)
            task_id_repeated = torch.from_numpy(np.array(range(len(x[0]))))
            task_id_repeated = task_id_repeated.repeat(len(x), 1).to(device)
            task_id_batch = task_id_repeated[:len(candidates)].to(device)
            output, attentions, task_embedding, encoder_output = model(candidates, task_id_batch)

            # 加权选出下轮新加train
            probs = [np.exp(high_gain_preference * o) for o in output]
            sum_prob = sum(probs)
            probs = [prob / sum_prob for prob in probs]

            selected_train_set = random.choices(candidates, probs, k=new_train_per_iter)
            for x in selected_train_set:
                in_test_list[binList2Int(x)] = False
            if (k * task_num + j) % mtgnet_upd_freq == mtgnet_upd_freq - 1:
                # model, min_loss = update_model(model, train_x, train_y, train_index)
                mtg_training(model, ensemble_num, dataset_name, gpu_id, step=1, end_num=1, dropout_rate=0.5)
