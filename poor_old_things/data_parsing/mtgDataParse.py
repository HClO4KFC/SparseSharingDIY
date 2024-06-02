import pandas as pd
import numpy as np
import copy
import torch

from utils.errReport import CustomError


class ParsedDataset:
    def __init__(self, dataset, device, x, y, trn_samp_idx, eval_samp_idx, x_test_source, y_test_source):
        mask = copy.deepcopy(x)
        self.x_train = x[trn_samp_idx]
        self.y_train = y[trn_samp_idx]
        self.mask_train = mask[trn_samp_idx]

        self.x_eval = x[eval_samp_idx]
        self.y_eval = y[eval_samp_idx]
        self.mask_eval = mask[eval_samp_idx]
        if dataset == '27tasks':
            self.x_test = x_test_source
            self.y_test = y_test_source
            self.mask_test = copy.deepcopy(self.x_test)
        elif dataset == 'cityscapes':
            self.x_test = x_test_source
            self.y_test = y_test_source
            self.mask_test = copy.deepcopy(self.x_test)
        else:
            test_samp_idx = eval_samp_idx
            self.x_test = x_test_source[test_samp_idx]
            self.y_test = y_test_source[test_samp_idx]
            self.mask_test = copy.deepcopy(self.x_test)

        self.trn_size = len(trn_samp_idx)
        self.eval_size = len(eval_samp_idx)
        self.test_size = len(self.x_test)
        self.task_num = len(x[0])
        self.task_id_repeated = torch.from_numpy(np.array(range(len(x[0]))))
        self.task_id_repeated = self.task_id_repeated.repeat(len(x), 1).to(device)

        # 数组转张量
        self.x_train = torch.FloatTensor(self.x_train).to(device)
        self.y_train = torch.FloatTensor(self.y_train).to(device)
        self.mask_train = torch.FloatTensor(self.mask_train).to(device)
        self.x_eval = torch.FloatTensor(self.x_eval).to(device)
        self.y_eval = torch.FloatTensor(self.y_eval).to(device)
        self.mask_eval = torch.FloatTensor(self.mask_eval).to(device)
        self.x_test = torch.FloatTensor(self.x_test).to(device)
        self.y_test = torch.FloatTensor(self.y_test).to(device)
        self.mask_test = torch.FloatTensor(self.mask_test).to(device)

    def get_train_set(self):
        return self.x_train, self.y_train, self.mask_train

    def get_eval_set(self):
        return self.x_eval, self.y_eval, self.mask_eval

    def get_test_set(self):
        return self.x_test, self.y_test, self.mask_test


def get_dataset(dataset):
    print('loading', dataset)

    if dataset == 'mimic27':
        x = np.array(pd.read_csv('gain_collection/collected_gain/mimic27/mimic_x.csv', header=None, sep=' '))
        y = np.array(pd.read_csv('gain_collection/collected_gain/mimic27/mimic_y_valid.csv', header=None, sep=' '))
        testx = np.array(pd.read_csv('gain_collection/collected_gain/mimic27/mimic_x.csv', header=None, sep=' '))
        testy = np.array(
            pd.read_csv('gain_collection/collected_gain/mimic27/mimic_y_test.csv', header=None, sep=' '))
    elif dataset == 'cityscapes':
        x = np.array(pd.read_csv('gain_collection/collected_gain/cityscapes/trainVal_x.csv', header=None, sep=' '))
        y = np.array(pd.read_csv('gain_collection/collected_gain/cityscapes/trainVal_y.csv', header=None, sep=' '))
        testx = x
        testy = y
    else:
        raise CustomError('Cannot find dataset:' + str(dataset))
    return x, y, testx, testy


def data_shuffle(x, y, mask):
    shuffled_train_idx = torch.randperm(len(x))
    x = x[shuffled_train_idx]
    y = y[shuffled_train_idx]
    mask = mask[shuffled_train_idx]
    return x, y, mask


def data_slice(x, y, mask, i, bs):
    batch_from = bs * i
    batch_to = bs * (i + 1)
    batch_x = x[batch_from: batch_to]
    batch_y = y[batch_from: batch_to]
    batch_mask = mask[batch_from: batch_to]
    return batch_x, batch_y, batch_mask
