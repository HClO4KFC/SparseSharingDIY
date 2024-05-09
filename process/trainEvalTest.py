import os.path

import numpy as np
import torch
import pickle

from utils.dataParse import data_parse, data_shuffle
from model.netDef import HOINetTransformer
from trainDetails import train_base, update_ensemble
from evalDetails import eval_and_test


def model_training(dataset, ratio, temperature,
                   num_layers, num_hidden, ensemble_num,
                   gpu_id, step=1, end_num=1, seed=1,
                   strategy='active', dropout_rate=0.5):

    # 将用于保存最后一次训练中的任务嵌入信息和编码器层输出
    task_embedding_list = []
    encoder_output_list = []

    # 用于保存所有iter数据的数组们
    overall_train_loss = []
    overall_eval_loss = []
    overall_test_loss = []
    train_perf = np.array([])
    eval_perf = np.array([])
    test_perf = np.array([])
    overall_pred_traj = []
    overall_mask_traj = []

    save_no = 0
    save_path_pre = dataset + '_' + str(seed) + '_' + str(save_no)
    while True:
        if os.path.exists('./savings/embedding_collect/' + save_path_pre + '/'):
            save_no = save_no + 1
            save_path_pre = dataset + '_' + str(seed) + '_' + str(save_no)
        else:
            # os.mkdir('./savings/embedding_collect/' + save_path_pre + '/')
            break
    print("current step is", step)
    print("current strategy is", strategy)
    print("current seed is", seed)
    device = 'cuda:' + gpu_id
    print(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    suggested_end_num, parsed_data = data_parse(dataset, step, device)

    if end_num < 0:
        end_num = suggested_end_num
    # 默认参数为-1,意为按照建议截止批次进行

    if end_num < suggested_end_num:
        print('warning: end_num is suggested to be at least', suggested_end_num, ', now', end_num)
    else:
        print('the program will end after iter', end_num)

    # 设置训练参数
    train_batch_size = 128  # (训练过程中)每批处理这么多个分组方案x
    epoch_num = 10000  #
    criterion = torch.nn.L1Loss()  # 损失函数为各维度绝对差值的平均
    ensemble_capacity = ensemble_num
    max_patience = 5  # 若loss在连续这么多epoch内没有降低,则停止iter
    active_iter_num = 1000000  # TODO:为什么这么大

    # 进行数个训练iter,每个iter产生一个集成模型,进行一轮train-eval-test,使用一遍所有数据
    for active_iter in range(active_iter_num):
        print('# active_iter =', active_iter)
        is_last_iter = (active_iter == end_num)
        ensemble_list = []  # 集成模型中的基模型列表
        ensemble_loss = np.array([])  # 各个基模型的损失值,用于基模型的取舍
        iter_train_loss = np.array([])
        iter_eval_loss = np.array([])
        iter_test_loss = np.array([])
        patience_loss = 100  # 能容忍的最大损失函数值
        patience = 0  # 记录loss停止下降的epoch次数

        # 随机打乱训练集
        x_train, y_train, mask_train = data_shuffle(*parsed_data.get_train_set())

        # 初始化模型和优化器
        model = HOINetTransformer(
            num_layers=num_layers,
            model_dim=num_hidden,
            num_heads=1,
            task_dim=parsed_data.task_num,
            ffn_dim=num_hidden,
            dropout=dropout_rate
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.007,
            weight_decay=0.0005
        )

        # 采用集成学习方法,每个epoch在前面基础上训练同一个模型,并观察结果是否更好了
        print('epoch: ', end='')
        for epoch in range(epoch_num):
            if epoch % 10 == 9:
                print('*', end='')
            train_batch_num = int(len(x_train)/train_batch_size)
            # print('## epoch =', epoch, ', train', train_batch_size, 'x', train_batch_num, '.')

            epoch_output, epoch_gt, epoch_mask, epoch_encoder_output_list, epoch_task_embedding_list = train_base(
                model=model, optimizer=optimizer, criterion=criterion,
                train_batch_num=train_batch_num, batch_size=train_batch_size,
                parsed_data=parsed_data, is_last_iter=is_last_iter
            )
            task_embedding_list.extend(epoch_task_embedding_list)
            encoder_output_list.extend(epoch_encoder_output_list)

            ensemble_loss, epoch_train_loss = update_ensemble(
                model=model, criterion=criterion, epoch_output=epoch_output,
                epoch_gt=epoch_gt, epoch_mask=epoch_mask,
                ensemble_capacity=ensemble_capacity, ensemble_list=ensemble_list,
                ensemble_loss=ensemble_loss
            )

            # 将所有epoch的损失汇总到iter
            iter_train_loss = np.hstack((iter_train_loss, epoch_train_loss))

            # 若连续多个epoch的损失函数不降反增,则证明继续训练没有收益了,停止该iter
            if epoch_train_loss < patience_loss:
                patience = 0
                patience_loss = epoch_train_loss
            else:
                patience = patience + 1
                if patience == max_patience:
                    break
            # 验证测试阶段
            model.eval()

            # 设置验证参数
            eval_tst_batch_size = 25000
            ensemble_eval_output, ensemble_eval_gt, ensemble_eval_mask, \
                ensemble_test_output, ensemble_test_gt, ensemble_test_mask, \
                ensemble_encoder_output_list, ensemble_task_embedding_list, \
                x_eval, x_test, base_eval_mask, base_eval_gt, base_test_mask, base_test_gt = eval_and_test(
                    ensemble_list=ensemble_list, parsed_data=parsed_data,
                    batch_size=eval_tst_batch_size, device=device, end_num=end_num,
                    is_last_iter=is_last_iter, save_path_pre=save_path_pre)

            encoder_output_list.extend(ensemble_encoder_output_list)
            task_embedding_list.extend(ensemble_task_embedding_list)

            # 计算验证集上的损失
            # 比较双方: (除去所有不在mask中的空白格后)用所有基模型在所有eval数据上的输出取平均后与eval集的ground_truth相比较,比较方式由criterion给出
            ensemble_eval_loss = criterion(torch.mean(ensemble_eval_output, dim=0)[base_eval_mask != 0],
                                           base_eval_gt.mul(base_eval_mask)[base_eval_mask != 0]).cpu().detach().numpy()
            # 计算测试集上的总损失
            ensemble_test_loss = criterion(torch.mean(ensemble_test_output, dim=0)[base_test_mask != 0],
                                           base_test_gt.mul(base_test_mask)[base_test_mask != 0]).cpu().detach().numpy()

            # 将当前epoch构成的半成品集成模型的eval_loss数据汇总到iter
            iter_eval_loss = np.hstack((iter_eval_loss, ensemble_eval_loss))
            iter_test_loss = np.hstack((iter_test_loss, ensemble_test_loss))
        
        print()
        # 将当前iter的三阶段loss汇总起来
        overall_train_loss.append(iter_train_loss)
        overall_eval_loss.append(iter_eval_loss)
        overall_test_loss.append(iter_test_loss)

        overall_eval_output = torch.mean(ensemble_eval_output, dim=0)
        overall_test_output = torch.mean(ensemble_test_output, dim=0)

        # 计算并打印当前iter三个阶段结束时的loss
        train_perf = np.hstack((train_perf, epoch_train_loss))
        eval_perf = np.hstack((eval_perf, ensemble_eval_loss))
        test_perf = np.hstack((test_perf, ensemble_test_loss))
        print('train loss', train_perf)
        print('eval loss', eval_perf)
        print('test loss', test_perf)

        # 将当前iter的预测结果打包汇总,并备份训练数据
        if dataset == '27tasks':
            iter_pred = torch.cat([overall_test_output, overall_eval_output, y_train.cpu().detach()], 0).numpy()
            iter_x = torch.cat([x_test.cpu().detach(), x_eval.cpu.detach(), x_train.cpu().detach()], 0).numpy()
        else:
            iter_pred = torch.cat([overall_test_output, y_train.cpu().detach()], 0).numpy()
            iter_x = torch.cat([x_test.cpu().detach(), x_train.cpu().detach()], 0).numpy()
        overall_pred_traj.append(iter_pred)
        overall_mask_traj.append(iter_x)

        # 若iter轮次够了,保存log退出
        path = './log/'+dataset+'/'+ ratio +'/'
        if not os.path.exists(path):
            os.makedirs(path)
        if active_iter >= end_num:
            temp = strategy
            with open('./log/' + dataset + '/' + ratio + '/train_loss_' + temp + '_' + str(temperature) + '_' + str(seed) + '_' + str(num_layers)+'_'+str(active_iter)+'.pkl', "wb") as fp:
                pickle.dump(overall_train_loss, fp)
            with open('./log/'+dataset+'/'+ ratio +'/valid_loss_'+temp+'_'+str(temperature)+'_'+str(seed)+'_'+str(num_layers)+'_'+str(active_iter)+'.pkl', "wb") as fp:
                pickle.dump(overall_eval_loss, fp)
            with open('./log/'+dataset+'/'+ ratio +'/test_loss_'+temp+'_'+str(temperature)+'_'+str(seed)+'_'+str(num_layers)+'_'+str(active_iter)+'.pkl', "wb") as fp:
                pickle.dump(overall_test_loss, fp)
            with open('./log/'+dataset+'/'+ ratio +'/pred_traj'+temp+'_'+str(temperature)+'_'+str(seed)+'_'+str(num_layers)+'_'+str(active_iter)+'.pkl', "wb") as fp:
                pickle.dump(overall_pred_traj, fp)
            with open('./log/'+dataset+'/'+ ratio +'/mask_traj'+temp+'_'+str(temperature)+'_'+str(seed)+'_'+str(num_layers)+'_'+str(active_iter)+'.pkl', "wb") as fp:
                pickle.dump(overall_mask_traj, fp)
            print('The step for saving is', active_iter)
            return overall_pred_traj
