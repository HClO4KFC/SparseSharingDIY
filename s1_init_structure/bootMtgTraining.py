import os.path
import random
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from datasets.dataLoader import MultiDataset, collate_func
from dlfip.pytorch_object_detection.faster_rcnn.train_utils.train_eval_utils import train_one_epoch, evaluate
from dlfip.pytorch_segmentation.fcn.train_utils.train_and_eval import create_lr_scheduler, fcn_train_one_epoch, fcn_evaluate
from model.mtlModel import ModelTree
from s1_init_structure.mtg_net.transformer import HOINetTransformer
from poor_old_things.trainEvalTest import mtg_training
from utils.lookUpTables import get_init_lr


def try_mtl_train(train_loaders:list, val_loaders:list, backbone: str, grouping: list, out_features: list,
                  try_epoch_num: int, try_batch_num, print_freq:int, cv_tasks_args, lr=0.01, aux=False, amp=True, results_path_pre=None,
                  with_eval=False) -> list:
    # 输入: 一个分组方案grouping,其中在分组中的任务记作1,不在的记作0
    # 输出: 尝试将这些任务分为一组进行硬参数共享训练一定轮次后,得到各任务的loss向量
    member = [i for i in range(len(grouping)) if grouping[i] == 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelTree(
        backbone_name=backbone,
        member=member,
        out_features=out_features,
        prune_names=[],
        cv_tasks_args=cv_tasks_args,
        device=device).to(device)
    optims = []
    scheduler = []
    scalers = []
    for i in range(len(member)):
        params = [p for p in model.tasks[i].parameters() if p.requires_grad]
        if member[i] == 0:
            params_to_optimize = [
                {"params": [p for p in model.tasks[i].backbone.parameters() if p.requires_grad]},
                {"params": [p for p in model.tasks[i].classifier.parameters() if p.requires_grad]}
            ]

            if aux:
                params = [p for p in model.tasks[i].aux_classifier.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params, "lr": lr * 10})

            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=lr, momentum=0.9,
                weight_decay=1e-4
            )
            optims.append(optimizer)  # fcn optimizer
            scheduler.append(create_lr_scheduler(
                optimizer=optimizer,
                num_step=len(train_loaders[i]),
                epochs=30, warmup=True
            ))
            scalers.append(torch.cuda.amp.GradScaler() if amp else None)
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
            scalers.append(torch.cuda.amp.GradScaler() if amp else None)
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
                    data_loader=train_loaders[i],
                    device=device, epoch=epoch, print_freq=print_freq,
                    lr_scheduler=scheduler[ingroup_no],
                    scaler=scalers[ingroup_no],
                    try_batch_num=try_batch_num)
                train_loss[ingroup_no].append(mean_loss)
                learning_rate[ingroup_no].append(lr)
                if with_eval:
                    confmat = fcn_evaluate(
                        model=model.tasks[ingroup_no],
                        data_loader=val_loaders[i],
                        device=device, num_classes=21)
                    val_info = str(confmat)
                    print(f'val_info:{val_info}')
                    # write into txt
                    if results_path_pre is not None:
                        with open(os.path.join(results_path_pre, f'"{member}"grouped_{str(i)}th_result.txt'), "a") as f:
                            # 记录每个epoch对应的train_loss、lr以及验证集各指标
                            train_info = f"[epoch: {epoch}]\n" \
                                         f"train_loss: {mean_loss:.4f}\n" \
                                         f"lr: {lr:.6f}\n"
                            f.write(train_info + val_info + "\n\n")

                    save_file = {"model": model.tasks[ingroup_no].state_dict(),
                                 "optimizer": optims[ingroup_no].state_dict(),
                                 "lr_scheduler": scheduler[ingroup_no].state_dict(),
                                 "epoch": epoch}
                    if amp:
                        save_file["scaler"] = scalers[ingroup_no].state_dict()
                    torch.save(save_file, f'./save_weights/"{member}"grouped_{str(i)}th_model[{epoch}].pth')
            else:
                # calc loss of faster-rcnn
                mean_loss, lr = train_one_epoch(
                    model=model.tasks[ingroup_no],
                    optimizer=optims[ingroup_no],
                    data_loader=train_loaders[i],
                    device=device, epoch=epoch, print_freq=print_freq,
                    warmup=True, scaler=scalers[ingroup_no],
                    try_batch_num=try_batch_num)
                train_loss[ingroup_no].append(mean_loss.item())
                learning_rate[ingroup_no].append(lr)
                # update the learning rate
                scheduler[ingroup_no].step()
                if with_eval:
                    # evaluate on the test dataset
                    coco_info = evaluate(
                        model=model.tasks[ingroup_no],
                        data_loader=val_loaders[i],
                        device=device)

                    # write into txt
                    if results_path_pre is not None:
                        with open(os.path.join(results_path_pre, f'"{member}"grouped_{str(i)}th_result.txt'), "a") as f:
                            # 写入的数据包括coco指标还有loss和learning rate
                            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                            f.write(txt + "\n")

                    val_map.append(coco_info[1])  # pascal mAP

                    # save weights
                    save_files = {
                        'model': model.tasks[ingroup_no].state_dict(),
                        'optimizer': [optimizer.state_dict() for optimizer in optims],
                        'lr_scheduler': [sche.state_dict() for sche in scheduler],
                        'epoch': epoch}
                    if amp:
                        save_files["scaler"] = [scaler.state_dict() for scaler in scalers]
                    torch.save(save_files, f'./save_weights/"{member}"grouped_{str(i)}th_model[{epoch}].pth')

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
        x = int(x // 2)
    while len(ans) < lenth:
        ans.append(0)
    return ans


def mtg_active_learning(train_loaders, init_grp_args, mtgnet_args,
                        dataset_name, gpu_id, backbone, out_features, cv_task_args, val_loaders):

    mtgnet_upd_freq=init_grp_args.mtgnet_upd_freq
    high_gain_preference=init_grp_args.high_gain_prefer
    train_set_size=init_grp_args.train_set_develop * init_grp_args.meta_train_iter
    try_epoch_num=init_grp_args.labeling_try_epoch
    try_batch_num=init_grp_args.labeling_try_batch
    print_freq = init_grp_args.print_freq
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
        baseline_output = try_mtl_train(
            train_loaders=train_loaders, grouping=x, try_epoch_num=try_epoch_num,
            try_batch_num=try_batch_num, print_freq=print_freq,
            backbone=backbone, out_features=out_features,
            cv_tasks_args=cv_task_args, val_loaders=val_loaders)
        assert len([i for i in baseline_output if i is not None]) == 1
        baseline_y.append(sum([i for i in baseline_output if i is not None]))
        print(f'grouping: {x}, loss: {baseline_output}')
        # try_mtl_train()应接受一个0,1数组,在为1的任务中构建硬参数共享模型,试训练try_epoch_num个周期,记录并返回这段时间组内每个任务loss变化的斜率
        # 此外,为了debug用,应保存试训练期间的loss变化到文件
    all_x = []
    all_y = []
    print('开始主动学习...')
    for k in range(train_set_size // new_train_per_iter):
        print(f'迭代{str(k)}:')
        for j in range(task_num):
            print('  多任务试训练,标注一批元数据集:')
            # 制作上批选出的c_train的ground truth, 清空候选
            for new_member in selected_train_set:
                new_member_output = try_mtl_train(
                    train_loaders=train_loaders, val_loaders=val_loaders, try_epoch_num=try_epoch_num, try_batch_num=try_batch_num,
                    backbone=backbone, out_features=out_features, cv_tasks_args=cv_task_args, grouping=new_member, print_freq=print_freq)
                train_x.append(new_member)
                # train_y.append([new_member_output[i] - baseline_y[i] for i in range(task_num) if baseline_y[i] != 0])
                train_y.append(new_member_output)
                print(f'grouping: {new_member}, loss: {new_member_output}')
                all_x.append(new_member)
                all_y.append(new_member_output)
            print('  计算新一批数据集候选:')
            # 选出c_test中所有包含T_j的方案
            candidates = []
            for i in range(2 ** (task_num - 1) - 1):
                no = int2BinList(i, task_num - 1)
                no.insert(j, 1)
                assert len(no) == task_num
                num = binList2Int(no)
                if in_test_list[num]:
                    candidates.append(no)
            candidates_tensor = torch.Tensor(candidates).to(device)
            task_id_repeated = torch.from_numpy(np.array(range(task_num)))
            task_id_repeated = task_id_repeated.repeat(len(candidates_tensor), 1).to(device)
            task_id_batch = task_id_repeated[:len(candidates_tensor)].to(device)
            output, attentions, task_embedding, encoder_output = model(candidates_tensor, task_id_batch)
            output = output.detach().cpu().numpy().tolist()
            output = [[output[l][i] if candidates[l][i] != 0 else 0
                       for i in range(len(output[l]))]
                      for l in range(len(output))]
            assert len(candidates) == len(output)
            # for idx in range(len(candidates)):
            #     cand = candidates[idx]
            #     out = output[idx]
            #     print(f'grouping: {cand}, loss: {out}(predicted) ')


            # 加权选出下轮新加train
            probs = []
            for o in output:
                cnt = 0
                for item in o:
                    if item != 0:
                        cnt += 1
                probs.append(sum(o) / cnt)
            probs = [np.exp(high_gain_preference * prob) for prob in probs]
            probs = probs / sum(probs)
            selected_train_set = random.choices(candidates, probs.tolist(), k=new_train_per_iter)
            for x in selected_train_set:
                in_test_list[binList2Int(x)] = False
            if (k * task_num + j) % mtgnet_upd_freq == mtgnet_upd_freq - 1:
                print('  训练元学习模型:')
                # model, min_loss = update_model(model, train_x, train_y, train_index)
                model = mtg_training(
                    model=model, ensemble_num=ensemble_num, dataset_name=dataset_name,
                    gpu_id=gpu_id, step=1, end_num=1, trn_x=train_x, trn_y=train_y)
    return all_x, all_y, model
