import pickle

import numpy as np
import torch
import argparse
import os

from init_grouping.process.Root import mtg_training
from init_grouping.process.groupingSearch import mtg_beam_search
from init_grouping.utils.fileMgmt import load_models, load_parsed_data, save_models, save_parsed_data

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
    mtg_dataset = 'mimic27'
    step = 1

    # TODO:两两训练得出gain_collection(少训练几轮,用loss斜率判断最终loss会到达哪里)

    # 测试时尝试读取已有模型
    parsed_data = load_parsed_data(path_pre=save_path_pre, dataset=mtg_dataset, device='cuda'+gpu_id, step=step)
    mtg_ensemble_model = load_models(path_pre=save_path_pre, dataset=mtg_dataset, device='cuda'+gpu_id,
                                     end_num=mtg_end_num, ensemble_num=mtg_ensemble_size, gpu_id=gpu_id)

    # 用gain_collection训练MTG-net
    if (not testing) or parsed_data is None or mtg_ensemble_model is None:
        parsed_data, mtg_ensemble_model = mtg_training(
            dataset=mtg_dataset, ratio='1', temperature=0.00001,
            num_layers=2, num_hidden=128, ensemble_num=mtg_ensemble_size,
            gpu_id=gpu_id, step=step, end_num=mtg_end_num, seed=seed,
            strategy='active', dropout_rate=0.5)
        save_models(
            path_pre=save_path_pre, models=mtg_ensemble_model,
            dataset=mtg_dataset, device='cuda'+gpu_id, end_num=mtg_end_num)
        save_parsed_data(
            path_pre=save_path_pre, parsed_data=parsed_data, dataset=mtg_dataset,
            device='cuda'+gpu_id, step=step)

    # 根据MTG-net确定初始共享组的划分(小规模:枚举;大规模:波束搜索beam-search)
    print('finish the init grouping with beam-search method...')
    grouping = mtg_beam_search(parsed_data, mtg_ensemble_model, 'cuda:' + gpu_id, mtg_beam_width)

    # TODO: 建立分组稀疏参数共享模型(读adashare, sparseSharing)

    # TODO: (异步)监控任务表现,若有loss飘升,将该任务编号投入重训练队列

    pass
