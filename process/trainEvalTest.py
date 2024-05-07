import os.path

import numpy as np
import torch
import copy
import pickle

from utils.dataParse import data_parse, data_shuffle, data_slice
from model.netDef import HOINetTransformer
from trainDetails import train_base, update_ensemble


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
    save_path_pre = dataset + '_' + str(seed) + '_' + save_no
    while True:
        if os.path.exists('./savings/embedding_collect/' + save_path_pre + '/'):
            save_no = save_no + 1
            save_path_pre = dataset + '_' + str(seed) + '_' + save_no
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

    if end_num < suggested_end_num:
        print('warning: end_num is suggested to be at least', end_num)

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
        for epoch in range(epoch_num):
            train_batch_num = int(len(x_train)/train_batch_size)
            print('## epoch =', epoch, ', train', train_batch_size, 'x', train_batch_num, '.')

            epoch_output, epoch_gt, epoch_mask, epoch_encoder_output_list, epoch_task_embedding_list = train_base(
                model=model, optimizer=optimizer, criterion=criterion,
                train_batch_num=train_batch_num, batch_size=train_batch_size,
                parsed_data=parsed_data, is_last_iter=(active_iter == end_num)
            )

            ensemble_loss, epoch_train_loss = update_ensemble(
                model=model, criterion=criterion, epoch_output=epoch_output,
                epoch_gt=epoch_gt, epoch_mask=epoch_mask,
                ensemble_capacity=ensemble_capacity, ensemble_list=ensemble_list,
                ensemble_loss=ensemble_loss
            )

            # 将所有epoch的损失汇总到iter
            iter_train_loss = np.hstack(iter_train_loss, epoch_train_loss)

            # 若连续多个epoch的损失函数不降反增,则证明继续训练没有收益了,停止该iter
            if epoch_train_loss < patience_loss:
                patience = 0
                patience_loss = epoch_train_loss
            else:
                if patience == 0:
                    tmp_model = copy.deepcopy(model)
                patience = patience + 1
                if patience == max_patience:
                    break
            # 验证测试阶段
            model.eval()

            # 设置验证参数
            eval_tst_batch_size = 25000
            x_eval, y_eval, mask_eval = parsed_data.get_eval_set()
            ensemble_eval_output = torch.Tensor([])
            ensemble_test_output = torch.Tensor([])
            ensemble_eval_mask = torch.Tensor([])
            ensemble_test_mask = torch.Tensor([])

            # 验证各基模型在验证集上的表现
            for base_no in range(len(ensemble_list)):
                eval_batch_num = int(len(x_eval)/eval_tst_batch_size)
                print('evaluating base model no.', base_no, ', sized', eval_tst_batch_size, 'x', eval_batch_num, '.')
                base_model = ensemble_list[base_no]
                base_eval_output = torch.Tensor([])
                base_eval_g_truth = torch.Tensor([])
                base_eval_mask = torch.Tensor([])
                # 分批验证
                for batch in range(eval_batch_num + 1):
                    print('evaluating batch', batch)
                    # 切出本批次的x,y和mask
                    batch_x, batch_y, batch_mask = data_slice(x_eval, y_eval, mask_eval, batch, eval_tst_batch_size)
                    task_id_batch = parsed_data.task_id_repeated[:len(batch_x)].to(device)
                    if len(batch_x) == 0:
                        break

                    # 跑一批验证集数据
                    output, attentions, task_embedding, encoder_output = base_model(batch_x, task_id_batch)
                    if base_no == 0:
                        encoder_output = torch.mul(encoder_output.permute(2, 0, 1), batch_x).permute(1, 2, 0)
                        if active_iter == end_num:
                            task_embedding_list.append(task_embedding.cpu().detach().numpy())
                            encoder_output_list.append(encoder_output.cpu().detach().numpy())
                            if not os.path.exists('./savings/embedding_collect/' + save_path_pre + '/'):
                                os.mkdir('./savings/embedding_collect/' + save_path_pre + '/')
                            np.save('./savings/embedding_collect/' + save_path_pre +
                                    '/task_embedding_list_eval_' + str(base_no) + '_' + str(end_num))
                            np.save('./savings/embedding_collect/' + save_path_pre +
                                    '/encoder_output_list_eval_' + str(base_no) + '_' + str(end_num))
                    output = output.mul(batch_mask)
                    base_eval_output = torch.cat([base_eval_output, output.cpu().detach()], 0)
                    base_eval_g_truth = torch.cat([base_eval_g_truth, batch_y.cpu().detach()], 0)
                    base_eval_mask = torch.cat([base_eval_mask, batch_mask.cpu().detach()], 0)

                # 将所有基模型的eval阶段输出汇总到ensemble
                if len(ensemble_eval_output) == 0:
                    ensemble_eval_output = base_eval_output.clone().unsqeeze(dim=0)
                else:
                    ensemble_eval_output = torch.cat([ensemble_eval_output, base_eval_output.unsqueeze(dim=0)])

                # test阶段:
                base_test_output = torch.Tensor([])
                base_test_g_truth = torch.Tensor([])
                base_test_mask = torch.Tensor([])
                x_test, y_test, mask_test = parsed_data.get_test_set()
                test_batch_num = int(len(x_test) / eval_tst_batch_size)
                print('testing base model no.', base_no, ', sized', eval_tst_batch_size, 'x', test_batch_num, '.')
                for batch in range(+1):
                    print('testing batch', batch)
                    batch_x, batch_y, batch_mask = data_slice(x_test, y_test, mask_test, batch, eval_tst_batch_size)
                    if len(batch_x) == 0:
                        break
                    # 用训练好的基模型跑一批test数据
                    task_id_batch = parsed_data.task_id_repeated[:len(batch_x)].to(device)
                    output, attentions, _, _ = base_model(batch_x, task_id_batch)
                    output = output.mul(batch_mask)
                    # 保存测试阶段该基模型在所有测试输入上的输出,gt和mask
                    base_test_output = torch.cat([base_eval_output, output.cpu().detach()], 0)
                    base_test_g_truth = torch.cat([base_test_g_truth, batch_y.cpu().detach()], 0)
                    base_test_mask = torch.cat([base_test_mask, batch_mask.cpu().detach()], 0)

                # 将所有基模型在测试阶段的输出数据集中到ensemble
                if len(ensemble_test_output) == 0:
                    ensemble_test_output = base_test_output.clone().unsqeeze(dim=0)
                else:
                    ensemble_test_output = torch.cat([ensemble_test_output, base_test_output])
            # 总结:至此已经完成了本次epoch的分批训练和分批验证,接下来截止到本epoch,组成的集成模型loss情况

            # 计算验证集上的损失
            # 比较双方: (除去所有不在mask中的空白格后)用所有基模型在所有eval数据上的输出取平均后与eval集的ground_truth相比较,比较方式由criterion给出
            ensemble_eval_loss = criterion(torch.mean(ensemble_eval_output, dim=0)[base_eval_mask != 0], base_eval_g_truth.mul(base_eval_mask)[base_eval_mask != 0]).cpu().detach().numpy()
            # 计算测试集上的总损失
            ensemble_test_loss = criterion(torch.mean(ensemble_test_output, dim=0)[base_test_mask != 0], base_test_g_truth.mul(base_test_mask)[base_test_mask != 0]).cpu().detach().numpy()

            # 将当前epoch构成的半成品集成模型的eval_loss数据汇总到iter
            iter_eval_loss = np.hstack(iter_eval_loss, ensemble_eval_loss)

        # 将当前iter的三阶段loss汇总起来
        overall_train_loss.append(iter_train_loss)
        overall_eval_loss.append(iter_eval_loss)
        overall_test_loss.append(iter_test_loss)

        overall_eval_output = torch.mean(ensemble_eval_output, dim=0)
        overall_test_output = torch.mean(ensemble_test_output, dim=0)

        # 计算并打印当前iter三个阶段结束时的loss
        train_perf = np.hstack((train_perf, epoch_train_loss))
        eval_perf = np.hstack((eval_perf, ensemble_eval_loss))
        test_perf = np.hdtack((test_perf, ensemble_test_loss))
        print('train loss', train_perf)
        print('eval loss', eval_perf)
        print('test loss', test_perf)

        # 将当前iter的预测结果打包汇总,并备份训练数据
        if dataset == '27tasks':
            iter_pred = torch.cat([overall_test_output, overall_eval_output, y_train.cpu().detach()], 0).numpy()
            iter_x = torch.cat([x_test.cpu().detach(), x_eval.cpu.detach(), x_train.cpu().detach()])
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
