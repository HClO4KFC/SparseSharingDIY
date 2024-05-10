import os.path
import pickle
import numpy as np
import torch
import argparse

from init_grouping.utils.dataParse import ParsedDataset
from init_grouping.process.trainEvalTest import mtg_training


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic27')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--ratio', type=str, default='1')
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--strategy', type=str, default='active')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--ensemble', type=int, default=2)
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--end_num', type=int, default=-1)
    in_args = parser.parse_args()
    return in_args


if __name__ == '__main__':
    args = get_args()
    # hyper-parameters
    dataset = args.dataset
    strategy = args.strategy
    # strategy = 'pertask' # pertask or perstep
    temperature = 0.00001
    ratio = args.ratio
    learning_rate = 0.1
    num_layers = args.layer_num
    gpu_id = args.gpu_id
    device = 'cuda' + gpu_id
    ensemble_num = args.ensemble
    num_hidden = args.num_hidden
    seed = args.seed
    step = args.step
    end_num = args.end_num
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parsed_data, best_ensemble_list = mtg_training(
        dataset=dataset, ratio=ratio, temperature=temperature, num_layers=num_layers,
        num_hidden=num_hidden, ensemble_num=ensemble_num, gpu_id=gpu_id, end_num=end_num,
        step=step, strategy=strategy, seed=seed, dropout_rate=args.dropout_rate)
    save_path_pre = 'test_files/'
    if not os.path.exists(save_path_pre + 'trained_models/'):
        os.makedirs(save_path_pre + 'trained_models/')
    if not os.path.exists(save_path_pre + 'parsed_data/'):
        os.makedirs(save_path_pre + 'parsed_data/')
    trained_model_path_pre = save_path_pre + 'trained_models/' + dataset + '_' + device + '_' + str(end_num) + '/'
    parsed_data_path = save_path_pre + 'parsed_data/' + dataset + '_' + device + '_' + str(step) + '.pkl'
    if not os.path.exists(trained_model_path_pre):
        print('models saving into', trained_model_path_pre + '...')
        os.mkdir(trained_model_path_pre)
        for i in range(len(best_ensemble_list)):
            base_model = best_ensemble_list[i]
            torch.save(base_model, trained_model_path_pre + 'model_' + str(i) + '.pth')
    else:
        print('not saving models, for dir ' + trained_model_path_pre + ' already exist.')
    if not os.path.exists(parsed_data_path):
        print('parsed data saving into', parsed_data_path + '...')
        with open(parsed_data_path, "wb") as f:
            pickle.dump(parsed_data, f)
    else:
        print('not saving models, for file ' + parsed_data_path + ' already exist.')
