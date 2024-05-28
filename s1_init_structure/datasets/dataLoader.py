import json
import os
import torch
import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.errReport import CustomError


def get_sub_item(file_name, args):
    ans_tensor = None
    if args.name == 'left' \
            or args.name == 'right' \
            or args.name == 'disparity' \
            or args.name == 'rain':
        # 读取单/三通道图片: leftImg8bit为左眼相机原图, rightImg8bit为右眼相机原图
        # image = Image.open(file_name).convert('RGB')
        # trans = transforms.Compose([transforms.ToTensor()])  # 定义数据预处理模式
        ans_tensor = Image.open(file_name).convert('RGB')
    elif args.name == 'instance' \
            or args.name == 'label' \
            or args.name == 'panoptic':
        # 读取像素色值形式的标记: labelIds为类别标记, instance为实例标记, 均用像素色值编码
        # trans = transforms.Compose([transforms.ToTensor()])  # 定义数据预处理模式
        ans_tensor = Image.open(file_name)
    elif args.name == 'people':
        with open(file_name, 'r') as f:
            data = json.load(f)

        objects = data['objects']
        boxes = []
        labels = []

        for obj in objects:
            bbox = obj['bbox']
            label = obj['label']
            boxes.append(bbox)
            labels.append(label)
        return boxes, labels
    elif args.name == 'car':
        with open(file_name, 'r') as f:
            data = json.load(f)

        # 提取2D对象信息
        objects = data['objects']
        boxes_2d = []
        labels = []
        for obj in objects:
            bbox_2d = obj['2d']['modal']  # 使用modal 2D边界框
            label = obj['label']
            boxes_2d.append(bbox_2d)
            labels.append(label)
        return boxes_2d, labels
    elif args.name == 'obj':
        with open(file_name, 'r') as f:
            data = json.load(f)
        objects = data['objects']
        bboxes_2d = []
        labels = []
        for obj in objects:
            bboxes_2d.append(obj['bbox'])
            labels.append(obj['label'])
        return bboxes_2d, labels
    else:
        raise CustomError("Unknown set_name='" + args.set_name + "' while loading datasets")
    return ans_tensor


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


class MultiDataset(Dataset):
    def __init__(self, dataset:str, path_pre:str, cv_tasks_args, cv_subsets_args, train_val_test:str, transform):

        # 任务和子集的名字和数量
        self.task_name = [cv_task_arg.name for cv_task_arg in cv_tasks_args]
        self.subset_name = [cv_subset_arg.name for cv_subset_arg in cv_subsets_args]
        self.task_num = len(self.task_name)
        self.subset_num = len(self.subset_name)

        # 方便起见,体内留存任务和子集的参数
        self.tasks_args = cv_tasks_args
        self.subsets_args = cv_subsets_args

        # 读取子集文件列表
        self.file_list = [gen_file_list(dataset, path_pre, self.subsets_args[i], train_val_test) for i in range(len(self.subset_name))]
        self.file_list_len = len(self.file_list[0])
        for i in range(self.task_num):
            assert len(self.file_list[i]) == self.file_list_len

        # 数据变换
        self.transforms = [(transform if cv_subsets_args[i].ext == 'png' else transforms.Compose([])) for i in range(self.subset_num)]
        self.transforms[self.subset_name.index('left')] = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return self.file_list_len

    def __getitem__(self, index):
        file_names = [self.file_list[i][index] for i in range(self.subset_num)]  # 每个subset当前index文件名
        sub_items = [get_sub_item(file_names[i], self.subsets_args[i]) for i in range(self.subset_num)]  # 按文件名取得所存数据
        sub_out = [self.transforms[i](sub_items[i]) for i in range(self.subset_num)]
        return sub_out, self.subset_name


class SingleDataset(Dataset):
    def __init__(self, dataset: str, path_pre: str, cv_task_arg, cv_subsets_args, train_val_test: str, transform):
        super(SingleDataset, self).__init__()
        # 找到对应的文件夹
        # 注: 此处cv_task_arg仅包含对应任务的arg
        self.data_args = [cv_subsets_args[i] for i in range(len(cv_subsets_args)) if cv_subsets_args[i].name == cv_task_arg.input][0]
        self.label_args = [cv_subsets_args[i] for i in range(len(cv_subsets_args)) if cv_subsets_args[i].name == cv_task_arg.output][0]
        self.cv_task_name = cv_task_arg.name

        self.data_file_list = gen_file_list(dataset, path_pre, self.data_args, train_val_test)
        self.label_file_list = gen_file_list(dataset, path_pre, self.label_args, train_val_test)
        assert len(self.data_file_list) == len(self.label_file_list)
        self.length = len(self.data_file_list)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_file_name = self.data_file_list[index]
        label_file_name = self.label_file_list[index]
        data_item = get_sub_item(data_file_name, self.data_args)
        label_item = get_sub_item(label_file_name, self.label_args)
        if self.transform is not None:
            data_item = self.transform(data_item)
            label_item = self.transform(label_item)
        return data_item, label_item


def collate_func(batch:list):
    # Initialize a dictionary to store the batched data
    batch_dict = {batch[0][1][subset]:[batch[no][0][subset] for no in range(len(batch))] for subset in range(len(batch[0][0]))}
    # Convert lists to tensors if necessary
    for subset_name, sub_items in batch_dict.items():
        batch_dict[subset_name] = [torch.stack(items) for items in sub_items]
    return batch_dict
