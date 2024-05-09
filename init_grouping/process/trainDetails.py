import torch
import copy
import numpy as np

from init_grouping.utils.dataParse import data_slice, ParsedDataset


def train_one_batch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion, parsed_data: ParsedDataset, batch: int, batch_size: int):
    # 切片出一批数据
    optimizer.zero_grad()
    batch_x, batch_y, batch_mask = data_slice(*(parsed_data.get_train_set()), batch, batch_size)
    if len(batch_x) == 0:
        return
    task_id_batch = parsed_data.task_id_repeated[:len(batch_x)]  # 将所有序号重复了batch_size遍

    # 前向传播
    output, attentions, task_embedding, encoder_output = model(batch_x, task_id_batch)
    # 输入是一批x,形状为(size, 27), 及每组标号(也是(size, 27)此处每行都是0-26,只是为了格式化输入)
    # output: 模型输出,即为train完了这步后的y^, 形状为(size, 27)
    # attentions: 注意力权重 TODO: 不知道干嘛用
    # task_embedding: 该次循环时27个任务的编码(由于是动态图,编码随着训练进程会变化)
    #   形状为(27, 128''), 因为每行都一样所以只取第0行
    # encoder_output: 该批次中,经过公共encoder层后的输出,形状为(size, 27, 128)

    # 将编码器输出与分组方案(1001...11)相乘,也就是不在分组方案内的任务不重要,乘0以滤除影响
    encoder_output = torch.mul(encoder_output.permute(2, 0, 1), batch_x).permute(1, 2, 0)  # TODO:保存这个干嘛

    # 保存最后一轮iter中的encoder_output和task_embedding信息
    # if is_last_iter:
    #     encoder_output_list.append(encoder_output.cpu().detach().numpy())
    #     task_embedding_list.append(task_embedding.cpu().detach().numpy())
    #     if not os.path.exists('./savings/embedding_collect/' + save_path_pre + '/'):
    #         os.mkdir('./savings/embedding_collect/' + save_path_pre + '/')
    #     np.save ('./savings/embedding_collect/' + save_path_pre + '/' + 'task_embedding_list.npy', task_embedding_list)
    #     np.save ('./savings/embedding_collect/' + save_path_pre + '/' + 'encoder_output_list.npy', encoder_output_list)
    # (用当前批训练)计算当前batch损失函数,反向传播,更新参数
    loss = criterion(output[batch_mask != 0], batch_y[batch_mask != 0])
    loss.backward()
    optimizer.step()
    return output.cpu().detach(), batch_y.cpu().detach(), batch_mask.cpu().detach(), \
        encoder_output.cpu().detach().numpy(), task_embedding.cpu().detach().numpy()
    # # 将该batch模型输出, mask和ground truth汇总给epoch保存
    # epoch_output = torch.cat([epoch_output, output.cpu().detach()], 0)
    # epoch_g_truth = torch.cat([epoch_g_truth, batch_y.cpu().detach()], 0)
    # epoch_mask = torch.cat([epoch_mask, batch_mask.cpu().detach()], 0)


def train_base(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               criterion, train_batch_num: int, batch_size: int,
               parsed_data: ParsedDataset, is_last_iter: bool):
    model.train()
    epoch_output = torch.Tensor([])  # 所有批次数据的总输出
    epoch_gt = torch.Tensor([])  # 所有批次的总ground truth
    epoch_mask = torch.Tensor([])
    epoch_encoder_output_list = []
    epoch_task_embedding_list = []

    # 分批train过程
    for batch in range(train_batch_num + 1):
        # print('training batch ' + str(batch) + '...')
        batch_output, batch_gt, batch_mask, batch_encoder_output, batch_task_embedding = train_one_batch(
            model=model, optimizer=optimizer, criterion=criterion,
            parsed_data=parsed_data, batch=batch, batch_size=batch_size
        )

        if is_last_iter:
            epoch_encoder_output_list.append(batch_encoder_output)
            epoch_task_embedding_list.append(batch_task_embedding)

        epoch_output = torch.cat([epoch_output, batch_output], 0)
        epoch_gt = torch.cat([epoch_gt, batch_gt], 0)
        epoch_mask = torch.cat([epoch_mask, batch_mask], 0)

    return epoch_output, epoch_gt, epoch_mask, \
        epoch_encoder_output_list, epoch_task_embedding_list


def update_ensemble(model: torch.nn.Module, criterion,
                    epoch_output: torch.Tensor, epoch_gt: torch.Tensor,
                    epoch_mask: torch.Tensor, ensemble_capacity: int,
                    ensemble_list: list, ensemble_loss: np.array):

    # 当前epoch所有batch训练完毕,计算总体损失,评估当前epoch的训练结果是否能被加入集成模型
    epoch_train_loss = criterion(epoch_output[epoch_mask != 0], epoch_gt.mul(epoch_mask)[epoch_mask != 0])
    if len(ensemble_list) < ensemble_capacity:
        # 若集成模型容量未满,则将当前epoch的模型直接加入集成模型
        tmp_model = copy.deepcopy(model)
        tmp_model.eval()
        ensemble_list.append(tmp_model)
        ensemble_loss = np.hstack((ensemble_loss, epoch_train_loss))
    else:
        # 若集成模型容量已满,仅保留其中loss最小的一些
        if epoch_train_loss < ensemble_loss.max():
            # 若当前epoch模型loss比集成模型中任意一个基模型小,替换之
            ensemble_loss[np.argmax(ensemble_loss)] = epoch_train_loss
            tmp_model = copy.deepcopy(model)
            tmp_model.eval()
            ensemble_list[np.argmax(ensemble_loss)] = tmp_model
    return ensemble_loss, epoch_train_loss
