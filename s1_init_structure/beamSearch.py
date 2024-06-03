import numpy as np
import torch.nn


def calc_batch(model: torch.nn.Module, device, batch_x: list)->np.array:
    group_nums = []  # 每个候选分组方案分了几个组
    expanded_batch_x = []
    max_group_size = 0
    for candidate in batch_x:
        cnt = max(candidate)  # 包含的分组数
        group_nums.append(cnt)
        for i in range(1, cnt + 1):
            temp = [1 if x == i else 0 for x in candidate]  # 拆出candidate中第i个分组,用于喂给元学习模型
            if sum(temp) > max_group_size:
                max_group_size = sum(temp)
            expanded_batch_x.append(temp)
    # list转tensor
    expanded_batch_x = torch.Tensor(expanded_batch_x).to(device)

    # print('input valid batch_size', len(expanded_batch_x), sum(group_nums))
    # 用空输入补齐批次
    while len(expanded_batch_x) % 28 != 0:
        expanded_batch_x = torch.cat((expanded_batch_x, torch.zeros((1, len(batch_x[0]))).to(device)), dim=0)

    # print('size of original batch is', len(batch_x))
    # print('size of training batch is', len(expanded_batch_x))
    # print('progress: ', max_group_size, '/', len(batch_x[0]))
    task_id_batch = torch.from_numpy(np.array(range(len(expanded_batch_x[0])))).repeat(len(expanded_batch_x), 1).to(device)
    output, _, _, _ = model(expanded_batch_x, task_id_batch)
    output = output * torch.Tensor(expanded_batch_x).to(device)
    result = []
    cnt = 0
    # print(len(batch_x), len(group_nums))
    for i in range(len(batch_x)):
        group_result = torch.tensor([0 for _ in range(len(batch_x[0]))]).to(device)
        for j in range(group_nums[i]):
            group_result = group_result + output[j + cnt]
        cnt = cnt + group_nums[i]
        if len(result) == 0:
            result = group_result.cpu().detach().numpy()
        else:
            result = np.vstack((result, group_result.cpu().detach().numpy()))
    return result


def mtg_beam_search(task_num, mtg_model, device, beam_width: int):
    beam_x = [[0 for _ in range(task_num)]]
    search_iter = 0
    while True:
        # print('# search iter', search_iter)
        # print('candidate num', len(beam_x))
        search_iter = search_iter + 1
        batch_x = []
        for candidate in beam_x:
            # batch_x.append(candidate)
            for i in [ii for ii, val in enumerate(candidate) if val == 0]:
                # candidate中尚未考虑的任务处标号为0, 此处取各个尚未考虑的任务
                for j in range(1, max(candidate) + 2):
                    # 组号为正整数,此处指将考虑的新任务放入之前的某个组,或新开一组
                    new_candi_x = candidate[:]
                    new_candi_x[i] = j
                    batch_x.append(new_candi_x)
        batch_y = calc_batch(mtg_model, device, batch_x)  # batch_size x task_num
        row_sum = np.sum(batch_y, axis=1)  # 求出总增益最大的前beam_width个candidate作为新的beam_x

        # 去重
        unique_rows = torch.unique(torch.Tensor(batch_x), dim=0)  # 去除下批搜索目标分组中的重复值
        unique_rows = [[int(i) for i in row] for row in unique_rows]  # torch_unique返回小数,组号需要转为整数,因此转换
        inverse_indices = [(batch_x.index(row) if row in batch_x else -1) for row in unique_rows]  # 非重复分组在batch_x中的序号(unique中返回的inverse_ind..不可靠)
        assert len([item for item in inverse_indices if item == -1]) == 0
        row_sum = [row_sum[i] for i in inverse_indices]  # 总增益列表
        batch_x = [batch_x[i] for i in inverse_indices]  # 方案列表
        # 求总增益最大的五个candidate成为新一轮beam_x
        # max_indices = np.argpartition(row_sum, -beam_width)[-min(beam_width, len(row_sum)):]
        sorted_indices = np.argsort(row_sum)
        top_n_indices = sorted_indices[-beam_width:] if beam_width > len(sorted_indices) else sorted_indices
        beam_x = [batch_x[ii] for ii in top_n_indices]
        beam_x = [[int(i) for i in row] for row in beam_x]
        beam_y = [row_sum[ii] for ii in top_n_indices]
        if not any(0 in row for row in beam_x):
            # for x in beam_x:
            #     print(x)
            # print('The overall gain of the top', beam_width, 'is:')
            # print(beam_y)
            final_result = beam_x[np.argmax(beam_y)]
            print('initiated hard grouping as', final_result, 'with a gain of', np.max(beam_y))
            return final_result
