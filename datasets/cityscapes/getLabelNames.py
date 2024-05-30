import glob
import json
import os

import omegaconf

from utils.errReport import CustomError


def gen_file_list(dataset, path_pre, args, train_val_test):
    if dataset == 'cityscapes':
        path_pre = os.path.join(path_pre, dataset, args.category, train_val_test)
        file_name_like = '*_' + args.set_name + '.' + args.ext
        # \cvDatasets\cityscape\gtFine\train\aachen\aachen_000000_000019_gtFine_color.png
    else:
        raise CustomError("dataset '" + dataset + "'is not supported.")
    path_like = os.path.join(path_pre, '**', file_name_like)
    # print("path located, searching for images like " + path_like)

    # 检索文件夹下所有符合条件的文件并读取
    file_list = glob.glob(path_like, recursive=True)
    return file_list


if __name__ == '__main__':
    args = omegaconf.OmegaConf.load('D:\\doing\\SparseSharingDIY\\yamls\\default.yaml')
    subset_names = ['people', 'car', 'obj']
    for name in subset_names:
        # print(name)
        label_set = set([])
        file_list = gen_file_list(
            dataset='cityscapes', path_pre='..\\cvDatasets',
            args=[args.cv_subsets_args[i] for i in range(len(args.cv_subsets_args))
                  if args.cv_subsets_args[i].name == name][0],
            train_val_test='train') \
        + gen_file_list(
            dataset='cityscapes', path_pre='..\\cvDatasets',
            args=[args.cv_subsets_args[i] for i in range(len(args.cv_subsets_args))
                  if args.cv_subsets_args[i].name == name][0],
            train_val_test='val')
        for no in range(len(file_list)):
            # print(f"{no}/{len(file_list)}")
            file_name = file_list[no]
            if name == 'people':
                with open(file_name, 'r') as f:
                    data = json.load(f)
                objects = data['objects']
                for obj in objects:
                    label_set.add(obj['label'])
            elif name == 'car':
                with open(file_name, 'r') as f:
                    data = json.load(f)
                # 提取2D对象信息
                objects = data['objects']
                for obj in objects:
                    label_set.add(obj['label'])
            elif name == 'obj':
                with open(file_name, 'r') as f:
                    data = json.load(f)
                objects = data['objects']
                labels = []
                for obj in objects:
                    label_set.add(obj['label'])
        print(name, f'[{",".join(label_set)}]')