import numpy as np
import torch
import argparse

from process.trainEvalTest import model_training


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic27')
    parser.add_argument('--gpu_id', type=str, default='0')  # uses gpu #0(used to be 1)
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
    ensemble_num = args.ensemble
    num_hidden = args.num_hidden
    seed = args.seed
    step = args.step
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    total_pred_traj = model_training(dataset=dataset, ratio=ratio, temperature=temperature, num_layers=num_layers,
                                     num_hidden=num_hidden, ensemble_num=ensemble_num, gpu_id=gpu_id, end_num=end_num,
                                     step=step, strategy=strategy, seed=seed, dropout_rate=args.dropout_rate)
