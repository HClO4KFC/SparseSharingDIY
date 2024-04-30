import os.path

import numpy as np
import torch
import copy

from dataParse import get_dataset
from netDef import HOINetTransformer

def model_training(dataset, ratio, temperature,
                   num_layers, num_hidden, ensemble_num,
                   gpu_id, step=1, end_num=1, seed=1,
                   strategy = 'active', dropout_rate = 0.5):

    # 保存最后一次训练中的任务嵌入信息和编码器层输出
    task_embedding_list = []
    encoder_output_list = []
    save_path_pre = './embedding_collect/' + dataset + '/'

    print("current step is",step)
    print("current strategy is",strategy)
    print("current seed is", seed)
    device = 'cuda:'+ gpu_id
    print(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x, y, x_test_source, y_test_source = get_dataset(dataset, ratio)
    # 从gain_collection中获得不同任务分组的ground truth增益

    task_id = torch.from_numpy(np.array(range(len(x[0])))).to(device)  # 用于为所有任务编号的张量,用to发送到训练设备,通常是cuda0
    task_id_repeated = task_id.repeat(len(x), 1)

    end_num = int((len(x[0]) * (len(x[0]) - 1)) / (2 * step))  # 确定训练主循环的次数
    if strategy == 'active':
        end_num = end_num + len(x[0]) - 1
    print("training process will stop after active_iter", end_num)
    active_iter_num = 1000000  # TODO:为什么这么大

    trn_samp_idx = []  # 训练集序号

    # 选择所有两两分组和全集分组方案加入训练集
    for i in range(len(x)):
        if len(x[i][x[i] == 1] == 1):  # 两两
            trn_samp_idx.append(i)
        if len(x[i][x[i] == 1]) == len(x[0]):  # 全集
            trn_samp_idx.append(i)
    val_samp_idx = np.setdiff1d(np.array(range(len(x))), trn_samp_idx) # 验证集是训练集的补集

    # 拆分训练集和验证集
    x_train = x[trn_samp_idx]
    y_train = y[trn_samp_idx]
    mask = copy.deepcopy(x)  # xx
    mask_train = mask[trn_samp_idx]  # xx
    # X_select = x[np.setdiff1d(np.array(range(len(x))), sample_list)]
    # y_select = y[np.setdiff1d(np.array(range(len(y))), sample_list)]
    # mask_select = mask[np.setdiff1d(np.array(range(len(mask))), sample_list)]

    # 拆分测试集
    if dataset != '27tasks':
        test_samp_idx = val_samp_idx
        x_test = x_test_source[test_samp_idx]
        y_test = y_test_source[test_samp_idx]
        mask_test = copy.deepcopy(x_test)  # xx
    else:
        x_test = x_test_source
        y_test = y_test_source
        mask_test = copy.deepcopy(x_test)  # xx

    # 数组转张量
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    # X_select = torch.FloatTensor(X_select)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    # y_select = torch.FloatTensor(y_select)
    mask_train = torch.FloatTensor(mask_train)  # xx
    mask_test = torch.FloatTensor(mask_test)  # xx
    # mask_select = torch.FloatTensor(mask_select)

    # 设置训练参数
    batch_size = 128  #
    epoch_num = 10000  #
    criterion = torch.nn.L1Loss()  # 损失函数为各维度绝对差值的平均
    ensemble_capacity = ensemble_num
    max_patience = 5

    # 模型实例化
    model = HOINetTransformer(num_layers=num_layers,
            model_dim=num_hidden,
            num_heads=1,
            task_dim = len(x[0]),
            ffn_dim= num_hidden,
            dropout=dropout_rate).to(device)

    # 开始训练
    for active_iter in range(active_iter_num):
        print('# active_iter =', active_iter)
        shuffled_train_idx = torch.randperm(len(x_train))
        x_train = x_train[shuffled_train_idx]
        y_train = y_train[shuffled_train_idx]
        mask_train = mask_train[shuffled_train_idx]  # xx
        model = HOINetTransformer(
            num_layers=num_layers,
            model_dim=num_hidden,
            num_heads=1,
            task_dim=len(x[0]),
            ffn_dim=num_hidden,
            dropout=dropout_rate
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.007,
            weight_decay=0.0005
        )
        for epoch in range(epoch_num):
            print('## epoch =', epoch)
            model.train()
            total_output = torch.Tensor([])  # 所有批次数据的总输出
            ground_truth = torch.Tensor([])  # 所有批次的总ground truth
            total_mask = torch.Tensor([])  # xx
            for batch in range(int(len(x_train)/batch_size)+1):
                # 切片出一批数据
                print('### batch =', batch)
                optimizer.zero_grad()
                batch_from = batch_size*batch
                batch_to = batch_size*(batch+1)
                batch_x = x_train[batch_from: batch_to].to(device)
                batch_y = y_train[batch_from: batch_to].to(device)
                if len(batch_x) == 0:
                    break  # 空批次
                batch_mask = mask_train[batch_from: batch_to].to(device)
                task_id_batch = task_id_repeated[:len(batch_x)]  # 将所有序号重复了batch_size遍

                # 前向传播
                output, attentions, task_embedding, encoder_output = model(batch_x, task_id_batch)
                # 输入是一批x,形状为(size, 27), 及每组标号(也是(size, 27)此处每行都是0-26,只是为了格式化输入)
                # output: 模型输出,即为train完了这步后的y^, 形状为(size, 27)
                # attentions: 注意力权重 TODO: 不知道干嘛用
                # task_embedding: 该次循环时27个任务的编码(由于是动态图,编码随着训练进程会变化)
                #   形状为(27, 128''), 因为每行都一样所以只取第0行
                # encoder_output: 该批次中,经过公共encoder层后的输出,形状为(size, 27, 128)

                # 将编码器输出与分组方案(1001...11)相乘,也就是不在分组方案内的任务不重要,乘0以滤除影响
                encoder_output = torch.mul(encoder_output.permute(2, 0, 1), batch_x).permute(1, 2, 0)

                # 保存最后一轮active_num训练中的encoder_output和task_embedding信息
                if active_iter == end_num:
                    encoder_output_list.append(encoder_output.cpu().detach().numpy())
                    task_embedding_list.append(task_embedding.cpu().detach().numpy())
                    times = 0
                    while True:
                        if os.path.exists(save_path_pre + str(seed) + '_' + str(times)):
                            times = times+1
                        else:
                            save_path_pre = save_path_pre + str(seed) + str(times)
                            os.mkdir(save_path_pre)
                            break
                    np.save(save_path_pre + '/' + 'task_embedding_list.npy', task_embedding_list)
                    np.save(save_path_pre + '/' + 'encoder_output_list.npy', encoder_output_list)

                # 记录模型输出和
                total_output = torch.cat([total_output, output.cpu().detach()],0)
                ground_truth = torch.cat([ground_truth, batch_y.cpu().detach()], 0)

                # 计算损失函数,反向传播,更新参数
                total_mask = torch.cat([total_mask, batch_mask.cpu().detach()], 0)
                loss = criterion(output[batch_mask != 0], batch_y[batch_mask != 0])
                loss.backward()
                optimizer.step()
            train_loss = criterion(total_output[total_mask != 0], ground_truth.mul(total_mask))
            if epoch < ensemble_capacity:
