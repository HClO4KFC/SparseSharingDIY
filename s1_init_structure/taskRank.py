import itertools

import numpy
import numpy as np
import torch.nn

from utils.errReport import CustomError


def mtg_task_rank(task_num:int, mtg_model:torch.nn.Module, device:str,
                  task_rank_args, grouping:list):
    if task_num < 2:
        raise CustomError('任务总数不足2,没有必要使用分组共享哦')
    max_iter = task_rank_args.max_iter
    rand_jump_rate = task_rank_args.rand_jump_rate
    converge_tolerance = task_rank_args.converge_tolerance
    indices = list(itertools.combinations(range(task_num), 2))
    input = torch.Tensor([[1 if i in pair else 0 for i in range(task_num)] for pair in indices])
    task_ids_repeated = torch.from_numpy(np.array(range(len(input[0])))).repeat(len(input), 1).to(device)
    output, _, _, _ = mtg_model(input, task_ids_repeated)

    # 获得邻接矩阵
    map = np.zeros((task_num, task_num))
    assert len(input) == len(output)
    for idx in range(len(input)):
        group = input[idx]
        gain = output[idx]
        members = [i for i in range(task_num) if group[i] == 1]
        assert len(members) == 2
        map[members[0]][members[1]] = gain[members[1]]
        map[members[1]][members[0]] = gain[members[0]]

    # 转成邻接矩阵,计算task rank
    adj_mat = map.T
    # 将无出链接的节点处理为均匀分布到所有节点
    out_degree = np.sum(adj_mat, axis=0)
    for i in range(task_num):
        if out_degree[i] == 0:
            adj_mat[:, i] = 1 / task_num
    trans = adj_mat / np.sum(adj_mat, axis=0)

    # 初始TaskRank向量
    task_ranks = numpy.ones(task_num) / task_num
    rand_jump = np.ones(task_num) / task_num
    for i in range(max_iter):
        tr_new = (1 - rand_jump_rate) * np.dot(trans, task_ranks) + rand_jump_rate * rand_jump
        if np.linalg.norm(tr_new - task_ranks, ord=1) < converge_tolerance:
            task_ranks = tr_new
            break
        task_ranks = tr_new

    main_tasks = np.zeros((np.max(grouping) + 1))
    max_tr = np.zeros((np.max(grouping) + 1))  # 第i组的最大tr
    for i in range(task_num):
        if task_ranks[i] > max_tr[grouping[i]]:
            max_tr[grouping[i]] = task_ranks[i]
            main_tasks[grouping] = i

    return task_ranks, main_tasks
