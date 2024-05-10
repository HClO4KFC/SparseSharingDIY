import pickle

import numpy as np
import torch
import argparse
import os

from init_grouping.process.Root import mtg_training
from init_grouping.process.groupingSearch import mtg_beam_search

global testing


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mtg_ensemble_size', type=int, default=2)
    parser.add_argument('--mtg_end_num', type=int, default=-1)
    parser.add_argument('--beam_width', type=int, default=20)
    parser.add_argument('--testing', type=str, default='0')
    parser.add_argument('--save_path_pre', type=str, default='test_files/')
    in_args = parser.parse_args()
    return in_args


if __name__ == '__main__':
    args = get_args()
    testing = True if args.testing == '1' else False
    gpu_id = args.gpu_id
    seed = args.seed
    mtg_ensemble_size = args.mtg_ensemble_size
    mtg_end_num = args.mtg_end_num
    mtg_beam_width = args.beam_width
    save_path_pre = args.save_path_pre
    dataset = 'mimic27'
    step = 1

    # TODO:两两训练得出gain_collection(少训练几轮,用loss斜率判断最终loss会到达哪里)

    # 用gain_collection训练MTG-net
    # 注:可讨论的参数:ensemble_num
    if not testing:
        parsed_data, mtg_ensemble_model = mtg_training(
            dataset=dataset, ratio='1', temperature=0.00001,
            num_layers=2, num_hidden=128, ensemble_num=mtg_ensemble_size,
            gpu_id=gpu_id, step=step, end_num=mtg_end_num, seed=seed,
            strategy='active', dropout_rate=0.5)
    else:
        trained_model_path_pre = save_path_pre + 'trained_models/' + dataset + '_' + 'cuda' + gpu_id + '_' + str(mtg_end_num) + '/'
        parsed_data_path = save_path_pre + 'parsed_data/' + dataset + '_' + 'cuda' + gpu_id + '_' + str(step) + '.pkl'
        if os.path.exists(save_path_pre + 'trained_models/') and os.path.exists(parsed_data_path):
            mtg_ensemble_model = []
            print('pretrained model and data found at', save_path_pre, ', loading...')
            for i in range(mtg_ensemble_size):
                base_model = torch.load(trained_model_path_pre + 'model_' + str(i) + '.pth').to('cuda:' + gpu_id)
                base_model.eval()
                mtg_ensemble_model.append(base_model)
            with open(parsed_data_path, 'rb') as f:
                parsed_data = pickle.load(f)
        else:
            print('pretrained model and data not found at "' + trained_model_path_pre + '" and "' + parsed_data_path + '", retraining...')
            parsed_data, mtg_ensemble_model = mtg_training(
                dataset=dataset, ratio='1', temperature=0.00001,
                num_layers=2, num_hidden=128, ensemble_num=mtg_ensemble_size,
                gpu_id=gpu_id, step=step, end_num=mtg_end_num, seed=seed,
                strategy='active', dropout_rate=0.5)

    # TODO: 根据MTG-net确定初始共享组的划分(小规模:枚举;大规模:波束搜索beam-search)
    print('finish the init grouping with beam-search method...')
    grouping = mtg_beam_search(parsed_data, mtg_ensemble_model, 'cuda:' + gpu_id, mtg_beam_width)

    # TODO: 建立分组稀疏参数共享模型(读adashare, sparseSharing)

    # TODO: (异步)监控任务表现,若有loss飘升,将该任务编号投入重训练队列

    pass
