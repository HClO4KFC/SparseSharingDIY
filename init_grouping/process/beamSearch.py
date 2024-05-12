import numpy as np
import torch.nn


def calc_batch(models: list, device: str, batch_x: list)->np.array:
    group_nums = []
    expanded_batch_x = []
    max_group_size = 0
    for candidate in batch_x:
        cnt = max(candidate)  # 包含的分组数
        group_nums.append(cnt)
        for i in range(1, cnt + 1):
            temp = [1 if x == i else 0 for x in candidate]
            if sum(temp) > max_group_size:
                max_group_size = sum(temp)
            expanded_batch_x.append(temp)
    # list转tensor
    expanded_batch_x = torch.Tensor(expanded_batch_x).to(device)

    # print('input valid batch_size', len(expanded_batch_x), sum(group_nums))
    # 用空输入补齐批次
    while len(expanded_batch_x) % 28 != 0:
        expanded_batch_x = torch.cat((expanded_batch_x, torch.zeros((1, 27)).to(device)), dim=0)

    # print('size of original batch is', len(batch_x))
    # print('size of training batch is', len(expanded_batch_x))
    # print('progress: ', max_group_size, '/', len(batch_x[0]))
    task_id_batch = torch.from_numpy(np.array(range(len(expanded_batch_x[0])))).repeat(len(expanded_batch_x), 1).to(device)
    ensemble_output = 0
    for base_model in models:
        base_output, _, _, _ = base_model(expanded_batch_x, task_id_batch)
        ensemble_output = ensemble_output + base_output
    ensemble_output = ensemble_output / len(models)
    ensemble_output = ensemble_output * torch.Tensor(expanded_batch_x).to(device)
    result = []
    cnt = 0
    # print(len(batch_x), len(group_nums))
    for i in range(len(batch_x)):
        group_result = torch.tensor([0 for _ in range(len(batch_x[0]))]).to(device)
        for j in range(group_nums[i]):
            group_result = group_result + ensemble_output[j + cnt]
        cnt = cnt + group_nums[i]
        if len(result) == 0:
            result = group_result.cpu().detach().numpy()
        else:
            result = np.vstack((result, group_result.cpu().detach().numpy()))
    return result


def mtg_beam_search(parsed_data, mtg_ensemble_model, device: str, beam_width: int):
    task_num = parsed_data.task_num
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
        batch_y = calc_batch(mtg_ensemble_model, device, batch_x)  # batch_size x task_num
        row_sum = np.sum(batch_y, axis=1)  # 求出总增益最大的前beam_width个candidate作为新的beam_x

        # 去重
        unique_rows = torch.unique(torch.Tensor(batch_x), dim=0)
        unique_rows = [[int(i) for i in row] for row in unique_rows]
        inverse_indices = [batch_x.index(row) if row in batch_x else -1 for row in unique_rows]
        row_sum = [row_sum[i] for i in inverse_indices]
        batch_x = [batch_x[i] for i in inverse_indices]
        # 求总增益最大的五个candidate成为新一轮beam_x
        max_indices = np.argpartition(row_sum, -beam_width)[-beam_width:]
        beam_x = [batch_x[ii] for ii in max_indices]
        beam_x = [[int(i) for i in row] for row in beam_x]
        beam_y = [row_sum[ii] for ii in max_indices]
        if not any(0 in row for row in beam_x):
            # for x in beam_x:
            #     print(x)
            # print('The overall gain of the top', beam_width, 'is:')
            # print(beam_y)
            final_result = beam_x[np.argmax(beam_y)]
            print('initiated hard grouping as', final_result, 'with a gain of', np.max(beam_y))
            return final_result
