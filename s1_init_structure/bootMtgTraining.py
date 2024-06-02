import random
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from datasets.dataLoader import MultiDataset, collate_func
from dlfip.pytorch_object_detection.faster_rcnn.train_utils.train_eval_utils import train_one_epoch
from dlfip.pytorch_segmentation.fcn.train_utils import create_lr_scheduler, fcn_train_one_epoch, evaluate
from model.mtlModel import ModelTree
from s1_init_structure.mtg_net.transformer import HOINetTransformer
from poor_old_things.trainEvalTest import mtg_training
from utils.lookUpTables import get_init_lr


def try_mtl_train(data_loaders, backbone: str, grouping: list, out_features: list,
                  try_epoch_num: int, cv_tasks_args, lr=0.01, aux=False) -> list:
    # 输入: 一个分组方案grouping,其中在分组中的任务记作1,不在的记作0
    # 输出: 尝试将这些任务分为一组进行硬参数共享训练一定轮次后,得到各任务的loss向量
    member = [i for i in range(len(grouping)) if grouping[i] == 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelTree(
        backbone_name=backbone,
        member=member,
        out_features=out_features,
        prune_names=[],
        cv_tasks_args=cv_tasks_args).to(device=device)
    optims = []
    scheduler = []
    for i in range(len(member)):
        params = [p for p in model.tasks[i].parameters() if p.requires_grad]
        if member[i] == 0:
            params_to_optimize = [
                {"params": [p for p in model.tasks[i].backbone.parameters() if p.requires_grad]},
                {"params": [p for p in model.tasks[i].classifier.parameters() if p.requires_grad]}
            ]

            if aux:
                params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params, "lr": lr * 10})

            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=lr, momentum=0.9,
                weight_decay=1e-4
            )
            optims.append(optimizer)  # fcn optimizer
            scheduler.append(create_lr_scheduler(
                optimizer=optimizer,
                num_step=len(data_loaders[i]),
                epochs=30, warmup=True
            ))
        else:
            optimizer = torch.optim.SGD(
                params=params,
                lr=lr,
                momentum=0.9,
                weight_decay=1e-4
            )
            optims.append(optimizer)
            scheduler.append(torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=3,
                gamma=0.33
            ))
    # criterions = [torch.nn.L1Loss() for i in member]  # buggy
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loss = [[] for _ in member]
    learning_rate = [[] for _ in member]
    val_map = [[] for _ in member]
    for epoch in range(try_epoch_num):
        model.train()
        for i in member:
            ingroup_no = member.index(i)
            if i == 0:
                # calc loss of fcn
                mean_loss, lr = fcn_train_one_epoch(
                    model=model.tasks[ingroup_no],
                    optimizer=optims[ingroup_no],
                    data_loader=data_loaders[i],
                    device=device, epoch=epoch, print_freq=1,
                    lr_scheduler=scheduler[ingroup_no], scaler=None)

                confmat = evaluate(
                    model=model, data_loader=data_loaders[i],
                    device=device, num_classes=21)
                val_info = str(confmat)
                print(val_info)
                train_loss[ingroup_no].append((mean_loss.item()))
                learning_rate[ingroup_no].append(lr)
            else:
                # calc loss of faster-rcnn
                mean_loss, lr = train_one_epoch(
                    model=model.tasks[ingroup_no],
                    optimizer=optims[ingroup_no],
                    data_loader=data_loaders[i],
                    device=device, epoch=epoch, print_freq=1,
                    warmup=True, scaler=None)
                train_loss[ingroup_no].append(mean_loss.item())
                learning_rate[ingroup_no].append(lr)
                # update the learning rate
                scheduler[ingroup_no].step()
    ans_loss = [train_loss[i][-1] for i in range(len(member))]
    ans = [None for _ in range(len(grouping))]
    for i in range(len(grouping)):
        if i in member:
            ans[i] = ans_loss[member.index(i)]
    return ans

    # for ingroup_no in range(len(member)):
    #     backbone_optim.zero_grad()
    #     heads_optims[ingroup_no].zero_grad()
    #     outputs = model(x=data, task_id=model.member[ingroup_no])  # BUGGY
    #     loss = criterions[ingroup_no](outputs[ingroup_no], targets[ingroup_no])
    #     loss.backward()
    #     heads_optims[ingroup_no].step()
    #     backbone_optim.step()
    #
    #     if batch_idx % 10 == 0:
    #         print(
    #             f"Epoch {epoch + 1}/{try_epoch_num}, Task {cv_task_no}, Batch {batch_idx + 1}/{len(multi_train_dataset)}, "
    #             f"Loss: {loss.item():.4f}")


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


def mtg_active_learning(dataloaders, init_grp_args, mtgnet_args,
                        dataset_name, gpu_id, backbone, out_features, cv_task_args):
    mtgnet_upd_freq=init_grp_args.mtgnet_upd_freq
    high_gain_preference=init_grp_args.high_gain_prefer
    train_set_size=init_grp_args.train_set_develop * init_grp_args.meta_train_iter
    try_epoch_num=init_grp_args.labeling_try_epoch
    new_train_per_iter=init_grp_args.train_set_develop
    num_layers=mtgnet_args.num_layers
    num_hidden=mtgnet_args.num_hidden
    dropout_rate=mtgnet_args.dropout_rate
    # device='cuda:' + gpu_id
    device = torch.device('cuda:'+gpu_id if torch.cuda.is_available() else 'cpu')
    ensemble_num=init_grp_args.ensemble_num
    task_num = len(cv_task_args)

    train_x = []
    train_y = []
    ensemble_model = []
    ensemble_model_loss = []
    in_test_list = [True for _ in range(2 ** (task_num + 1))]
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
    print('单任务训练中...')
    for x in baseline_x:
        x = (torch.Tensor(x)).to(device)
        baseline_output = try_mtl_train(data_loaders=dataloaders, grouping=x, try_epoch_num=try_epoch_num,
                                        backbone=backbone, out_features=out_features, cv_tasks_args=cv_task_args)
        assert len([i for i in baseline_output if i != 0]) == 1
        baseline_y.append(sum(baseline_output))
        print(f'grouping: {x}, loss: {baseline_output}')
        # try_mtl_train()应接受一个0,1数组,在为1的任务中构建硬参数共享模型,试训练try_epoch_num个周期,记录并返回这段时间组内每个任务loss变化的斜率
        # 此外,为了debug用,应保存试训练期间的loss变化到文件
    print('开始主动学习...')
    for k in range(train_set_size // new_train_per_iter):
        print(f'迭代过程{str(k)}:')
        for j in range(task_num):
            print('  多任务试训练,标注元数据集')
            # 制作上批选出的c_train的ground truth, 清空候选
            for new_member in selected_train_set:
                new_member_output = try_mtl_train(
                    data_loaders=dataloaders, try_epoch_num=try_epoch_num,
                    backbone=backbone, out_features=out_features, cv_tasks_args=cv_task_args,
                    grouping=new_member)
                train_x.append(new_member)
                # train_y.append([new_member_output[i] - baseline_y[i] for i in range(task_num) if baseline_y[i] != 0])
                train_y.append(new_member_output)
                print(f'grouping: {new_member}, loss: {new_member_output}')

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
                mtg_training(
                    model=model, ensemble_num=ensemble_num, dataset_name=dataset_name,
                    gpu_id=gpu_id, step=1, end_num=1, trn_x=train_x, trn_y=train_y)
