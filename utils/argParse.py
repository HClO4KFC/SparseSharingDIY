import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--testing', type=str, default='0')
    parser.add_argument('--save_path_pre', type=str, default='test_files/')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)

    # MTG net
    parser.add_argument('--mtg_ensemble_size', type=int, default=2)
    parser.add_argument('--mtg_end_num', type=int, default=-1)

    # MTG search
    parser.add_argument('--beam_width', type=int, default=20)

    # MTL models define
    parser.add_argument('--mtl_dataset_name', type=str, default='nyu_v2')
    parser.add_argument('--mtl_backbone_name', type=str, default='ResNet34')
    parser.add_argument('--out_features', type=str, default='')
    # TODO: figure out the out-put shape of each task in taskonomy, nyu_v2, DomainNet and cityscapes

    # single-task pruning
    parser.add_argument("--remain_percent", dest='remain_percent', type=float, default=0.1, help='percent of params to remain not to pruning')
    parser.add_argument("--max_pruning_iter", dest='max_pruning_iter', type=int, default=10, help='max times to pruning')
    parser.add_argument('--init_masks', dest='init_masks', type=str, default=None, help='initial masks for late reseting pruning')
    parser.add_argument('--need_cut', default='lstm,conv', type=str, dest='need_cut', help='parameters names that not cut')
    # parser.add_argument("--task_id", dest='task_id', type=int, default=0, help='the task to use')

    in_args, unk = parser.parse_known_args()
    if len(unk) != 0:
        print("warning: unknown args ", unk)
    return in_args


def str2list(s:str, conj:str)->list:
    lst = [item.strip() for item in s.split(conj) if item.strip()]
    return lst
