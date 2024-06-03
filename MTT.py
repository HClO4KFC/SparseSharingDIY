import argparse
import multiprocessing
import os
import pickle
import random
import sys
import time

import omegaconf
import numpy as np
import torch

from datasets.voc12.my_dataset import VOCDataSet
from dlfip.pytorch_segmentation.fcn.my_dataset import VOCSegmentation
from dlfip.pytorch_segmentation.fcn.train import get_transform, SegmentationPresetTrain, SegmentationPresetEval
from s2_continuous_sharing.MultiTaskTrain import multi_task_train
from s3_drift_mgmt.simTeacherModels import teachers
from train_utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from utils.lookUpTables import CvTask
from torchvision import transforms

from s1_init_structure.bootMtgTraining import mtg_active_learning, try_mtl_train
from s1_init_structure.beamSearch import mtg_beam_search
from datasets.dataLoader import MultiDataset
from s1_init_structure.taskRank import mtg_task_rank
from s2_continuous_sharing.rw_lock.RwLock import ReadWriteLock
from s2_continuous_sharing.prunerProc import pruner
from s3_drift_mgmt.async_proc.workerProc import worker
from s3_drift_mgmt.train_task_mgmt.trainTask import TrainTask
from s3_drift_mgmt.async_proc.trainManagerProc import train_manager
from model.mtlModel import ModelForrest, ModelTree
from s1_init_structure.initModelForrest import get_models


def get_obj_det_dl(task_no:int, batch_size:int):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # VOC数据集根目录
    VOC_root = os.path.join("dlfip", "pascalVOC")  # VOCdevkit
    aspect_ratio_group_factor = 3
    amp = False  # 是否使用混合精度训练，需要GPU支持

    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    class_map_path = os.path.join('class_maps', 'objdet' + str(task_no)+'.json')
    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(voc_root=VOC_root, year='2012',
                               class_map_path=class_map_path,
                               transforms=data_transform["train"],
                               txt_name=f"objdet{str(task_no)}_train.txt")
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=False,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, year="2012", transforms=data_transform["val"], txt_name=f"objdet{str(task_no)}_val.txt",
                             class_map_path=class_map_path)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)
    return train_dataset, val_dataset, train_data_loader, val_data_loader


def get_seg_dl(batch_size:int, data_path):

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    return train_dataset, val_dataset, train_loader, val_loader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='default.yaml')
    args = parser.parse_args()
    return omegaconf.OmegaConf.load(os.path.join('yamls', args.yaml))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def exp_multi_task_train(grouping, train_loaders, cv_tasks_args, val_loaders):
    try_epoch_num = 1#5
    try_batch_num = None
    print_freq = 1
    lr = 0.01
    aux = True
    amp = True
    with_eval = True
    print("正在进行单任务与多任务训练的对比...")
    # 单任务
    single_losses = [None for _ in range(len(grouping))]
    multi_losses = [None for _ in range(len(grouping))]
    start_time = time.time()
    print(f'single task start time: {start_time}')
    for i in range(1, len(grouping)):
        this_group = [1 if k == i else 0 for k in range(len(grouping))]
        print(f'单任务过程{this_group}:')
        loss = try_mtl_train(
            train_loaders=train_loaders, val_loaders=val_loaders,
            backbone='ResNet50', grouping=this_group, out_features=[],
            try_epoch_num=try_epoch_num, try_batch_num=try_batch_num,
            print_freq=print_freq, cv_tasks_args=cv_tasks_args,
            lr=lr, aux=aux, amp=amp, results_path_pre='./test1', with_eval=with_eval)
        single_losses = [loss[i] if this_group[i] == 1 else single_losses[i]]
    end_time = time.time()
    print(f'single task end time: {end_time}')
    print(f'single task total time{str(start_time - end_time)}')
    m_start_time = time.time()
    # for i in range(1, max(grouping) + 1):
    #     this_group = [1 if grouping[k] == i else 0 for k in range(len(grouping))]
    #     loss = try_mtl_train(
    #         train_loaders=train_loaders, val_loaders=val_loaders,
    #         backbone='ResNet50', grouping=this_group, out_features=[],
    #         try_epoch_num=try_epoch_num, try_batch_num=try_batch_num,
    #         print_freq=print_freq, cv_tasks_args=cv_tasks_args,
    #         lr=lr, aux=aux, amp=amp, results_path_pre='./test1', with_eval=with_eval)
    #     multi_losses = [loss[i] if this_group[i] == 1 else multi_losses[i]]
    m_end_time = time.time()
    pass


if __name__ == '__main__':
    args = get_args()
    basic_args = args.basic_args
    dataset_args = args.dataset_args
    mtgnet_args = args.mtgnet_args
    init_grp_args = args.init_grp_args
    beam_search_args = args.beam_search_args
    task_rank_args = args.task_rank_args
    mtl_design_args = args.mtl_design_args
    single_prune_args = args.single_prune_args
    multi_train_args = args.multi_train_args
    err_detect_args = args.err_detect_args
    mgmt_args = args.mgmt_args
    temp_args = args.temp_args
    cv_tasks_args = args.cv_tasks_args
    # cv_subsets_args = args.cv_subsets_args

    set_seed(0)
    # cal basic info
    task_num = 5
    batch_size = 1

    # 初始化任务描述(内置单任务数据集和加载器)
    # task_info_list = [CvTask(no=i, dataset_args=dataset_args,
    #                          cv_task_arg=cv_tasks_args[i],
    #                          cv_subsets_args=cv_subsets_args)
    #                   for i in range(task_num)]

    data_path = os.path.join('dlfip', 'pascalVOC')
    # 阶段1: 多任务模型构建
    # 准备多任务数据集
    train_sets = []
    val_sets = []
    train_loaders = []
    val_loaders = []

    # TODO:任务种类:语义分割
    seg_trn_set, seg_val_set, seg_trn_loader, seg_val_loader = get_seg_dl(batch_size=batch_size, data_path=data_path)
    train_loaders.append(seg_trn_loader)
    val_loaders.append(seg_val_loader)

    # TODO:任务种类: 目标检测
    # 图像预处理
    # 注：用RandomHorizontalFlip进行随机水平翻转后，ground truth 坐标也要翻转
    for task_no in range(1, 5):
        obj_det_trn_set, obj_det_val_set, obj_det_trn_loater, obj_det_val_loater = get_obj_det_dl(task_no=task_no, batch_size=batch_size)
        train_loaders.append(obj_det_trn_loater)
        val_loaders.append(obj_det_val_loater)
    # obj_det_trn_set, obj_det_val_set, obj_det_trn_loater, obj_det_val_loater = get_obj_det_dl(task_no=1)
    # for task_no in range(1, 5):
    #     train_loaders.append(obj_det_trn_loater)
    #     val_loaders.append(obj_det_val_loater)


    # multi_train_dataset = MultiDataset(
    #     dataset=dataset_args.dataset_name, path_pre=dataset_args.path_pre,
    #     cv_tasks_args=cv_tasks_args, cv_subsets_args=cv_subsets_args,
    #     train_val_test='train', transform=transforms.Compose([transforms.ToTensor()]),
    #     label_id_maps={cv_tasks_args[i].output:cv_tasks_args[i].label_id_map for i in range(len(cv_tasks_args)) if hasattr(cv_tasks_args[i], 'label_id_map')})
    # multi_val_dataset = MultiDataset(
    #     dataset=dataset_args.dataset_name, path_pre=dataset_args.path_pre,
    #     cv_tasks_args=cv_tasks_args, cv_subsets_args=cv_subsets_args,
    #     train_val_test='val', transform=transforms.Compose([transforms.ToTensor()]),
    #     label_id_maps={cv_tasks_args[i].output:cv_tasks_args[i].label_id_map for i in range(len(cv_tasks_args)) if hasattr(cv_tasks_args[i], 'label_id_map')})
    # 元数据集标注+元学习模型mtg-net训练(主动学习策略)
    # model = ModelTree(
    #     backbone_name='ResNet50',
    #     member=[0, 1],
    #     out_features=[],
    #     prune_names=[],
    #     cv_tasks_args=None
    # )
    # start_time = time.time()
    # all_x, all_y, meta_model = mtg_active_learning(
    #     train_loaders=train_loaders,
    #     val_loaders=val_loaders,
    #     init_grp_args=init_grp_args,
    #     mtgnet_args=mtgnet_args,
    #     dataset_name=dataset_args.dataset_name,
    #     gpu_id=basic_args.gpu_id,
    #     backbone=mtl_design_args.backbone,
    #     out_features=temp_args.out_features,
    #     cv_task_args=cv_tasks_args)
    #
    # # 波束搜索确定共享组的划分
    # print('finish the init_grouping with beam-search method...')
    # grouping = mtg_beam_search(
    #     task_num=task_num, mtg_model=meta_model,
    #     device=torch.device('cuda:' + basic_args.gpu_id if torch.cuda.is_available() else 'cpu'),
    #     beam_width=beam_search_args.beam_width)
    # end_time = time.time()
    # torch.save(meta_model.state_dict(), 'meta_model.pth')
    # save_dict = {}
    # save_dict['all_x'] = all_x
    # save_dict['all_y'] = all_y
    # save_dict['init_grp_args'] = init_grp_args
    # with open('meta_model_data.pkl', 'wb') as f:
    #     pickle.dump(save_dict, f)
    #
    grouping = [1, 3, 3, 3, 2]

    exp_multi_task_train(grouping=grouping, train_loaders=train_loaders, cv_tasks_args=cv_tasks_args, val_loaders=val_loaders)

    # try_mtl_train(
    #     train_loaders=train_loaders, val_loaders=val_loaders,
    #     backbone=mtl_design_args.backbone,
    #     grouping=grouping, out_features=[],
    #     try_epoch_num=init_grp_args.labeling_try_epoch,
    #     try_batch_num=init_grp_args.labeling_try_batch,
    #     print_freq=100, cv_tasks_args=cv_tasks_args,
    #     lr=0.01, aux=False, amp=True, results_path_pre=None,
    #     with_eval=True)

    # # pagerank获得任务重要性
    # task_ranks, main_tasks = mtg_task_rank(
    #     task_num=task_num, mtg_model=meta_model, device='cuda:' + basic_args.gpu_id,
    #     task_rank_args=task_rank_args, grouping=grouping)
    #
    # # 初始化参数共享模型(掩膜全通,为硬参数共享)
    # models = get_models(grouping=grouping, backbone_name=mtl_design_args.backbone,
    #                     prune_names=single_prune_args.need_cut,
    #                     out_features=temp_args.out_features, cv_task_args=cv_tasks_args)



    # 阶段2: 硬参数共享过渡到稀疏参数共享
    #
    # # 布置进程通信和互斥锁
    # dict_lock = ReadWriteLock()
    # manager = multiprocessing.Manager()
    # shared_dict = manager.dict()
    # for i in range(len(models)):
    #     shared_dict['model' + str(i)] = None
    #     shared_dict['mask' + str(i)] = None
    #
    # # 异步单任务剪枝
    # pruner_args = {
    #     'dict_lock': dict_lock, 'shared_dict': shared_dict,
    #     'task_ranks': task_ranks, 'single_prune_args': single_prune_args}
    # pruner = multiprocessing.Process(target=pruner, args=pruner_args)
    # pruner.start()
    # #
    # # # 多任务平行训练
    # multi_task_train(dict_lock=dict_lock, shared_dict=shared_dict,
    #                  multi_train_args=multi_train_args, cv_tasks_args=cv_tasks_args,
    #                  models=models, multi_train_dataset=multi_train_dataset,
    #                  task_info_list=task_info_list, task_ranks=task_ranks, grouping=grouping)
    # pruner.join()
    #
    # # 阶段3: 持续演化
    # # 启动训练管家进程
    # queue_to_train_manager = multiprocessing.Queue()
    # queue_from_train_manager = multiprocessing.Queue()
    # train_manager = multiprocessing.Process(target=train_manager, args=(
    #     mgmt_args.max_queue_lvl, mgmt_args.trainer_num, queue_to_train_manager, queue_from_train_manager))
    # train_manager.start()
    #
    # # 计算端设备上的所有任务-模型分派关系overall_allocation
    # task_allocation = [cv_task_arg.allocate for cv_task_arg in cv_tasks_args]
    # end_device_num = max(task_allocation)
    # assert end_device_num <= basic_args.available_end_device_num
    # overall_allocation = []  # end_device_num(fixed) * group_num(fixed) * group_size(vary)
    # for end_device_no in range(end_device_num):
    #     loads = [i for i in range(len(task_allocation)) if task_allocation[i] == end_device_no]
    #     group_no_of_loads = [grouping[load] for load in loads]
    #     grouped_loads = []
    #     for group_no in range(max(grouping)):
    #         list_ = [load for load in loads if grouping[load] == group_no]
    #         grouped_loads.append(list_)
    #     overall_allocation.append(grouped_loads)
    #
    # # 启动工作进程(模拟端设备)
    # queue_from_workers = multiprocessing.Queue()
    # queue_to_workers = []
    # workers = []
    # # 小模型备份
    # back_ups = []
    # for end_device_no in range(len(overall_allocation)):
    #     queue_to_workers.append(multiprocessing.Queue())
    #     allocation = overall_allocation[end_device_no]
    #     forrest_on_device = ModelForrest(
    #         tree_list=models, allocation=allocation,
    #         cv_subset_args=cv_subsets_args, cv_task_args=cv_tasks_args)
    #     back_ups.append(forrest_on_device)
    #     worker_args = {'forrest': forrest_on_device, 'worker_no': end_device_no,
    #                    'sample_interval': mgmt_args.sample_interval_sec,
    #                    'queue_from_main': queue_to_workers[end_device_no],
    #                    'queue_to_main': queue_from_workers}
    #     workers.append(multiprocessing.Process(
    #         target=worker,
    #         args=worker_args))
    #     workers[end_device_no].start()
    #
    # # 保存运行时数据集,用于重训练
    # # 分为len(workers)个子列表,分别存放每个端设备传回的最近data_buf_size个数据.子列表元素包含n+1位,前n位是subsets,最后一位是输入
    # runtime_data_buf = [[] for _ in range(end_device_num)]
    # # 开始持续演化,进程图worker(s)<=(<重分配)(抽样数据>)=>main<=(<重训结果)(重训任务>)=>train_manager<=(<重训结果)(重训任务)=>trainer(s)
    # error_cnt = [0 for _ in range(task_num)]
    # bias_cnt = [0 for _ in range(task_num)]
    # # runtime_dataset_size = err_detect_args.runtime_dataset_size
    # # runtime_dataset = [[None for _ in range(len(cv_subsets_args))] for _ in range(runtime_dataset_size)]
    # # runtime_dataset_idx = 0
    # models = ModelForrest(tree_list=models, allocation=[model.member for model in models], cv_subset_args=cv_subsets_args, cv_task_args=cv_tasks_args)
    # related_tasks_lists = [
    #     [task_id for task_id in range(task_num) if basic_args.distribution[task_id] == worker_no]
    #     for worker_no in range(end_device_num)]
    # # 封装边缘上分组共享模型群, 与每个end device上的forrest同构, 区别仅在于member多少
    # while True:
    #     if not queue_from_workers.empty():  # 收到来自某个端设备的采样数据
    #         new_message = queue_from_workers.get()
    #         # {'type': 'sample frame', 'sender': worker_no, 'data': data}
    #         msg_type = new_message['type']
    #         msg_sender = new_message['sender']
    #         msg_data = new_message['data']
    #         # {'input':[sub_out[i] for i in range(len(self.subset_name_list)) if self.subset_name_list[i] == 'rain'][0],
    #         #  'subsets':sub_out,
    #         #  'subset_names':self.subset_name_list}
    #         related_task_list = related_tasks_lists[msg_sender]
    #         targets = [teachers(task_id=task_id, msg_data=msg_data, cv_tasks_args=cv_tasks_args)
    #                    for task_id in related_task_list]  # 教师模型推理结果
    #         runtime_data_buf[msg_sender].append(targets)
    #         edge_output = models(msg_data['input'])
    #         edge_output_plain = [
    #             [edge_output[group_no][item_no]
    #              for group_no in range(len(edge_output))
    #              for item_no in range(len(edge_output[group_no]))
    #              if models.models[group_no].member[item_no] == task_id][0]
    #             for task_id in related_task_list
    #         ]
    #         end_output = back_ups[msg_sender](msg_data['input'])
    #         end_output_plain = [
    #             [end_output[group_no][item_no]
    #              for group_no in range(len(end_output))
    #              for item_no in range(len(end_output[group_no]))
    #              if models.models[group_no].member[item_no] == task_id][0]
    #             for task_id in related_task_list
    #         ]
    #         criterions = [task_info_list[rel_task].loss for rel_task in related_task_list]
    #         # 一段时间内,边缘模型群效果总好于端设备模型群,则更新端设备模型群
    #         sub_bias = [criterions[idx](end_output_plain[idx], edge_output_plain[idx]) for idx in range(len(related_task_list))]
    #         for list_idx in range(len(related_task_list)):
    #             task_id = related_task_list[list_idx]
    #             if sub_bias[list_idx] > err_detect_args.acc_threshold:
    #                 bias_cnt[task_id] += 1
    #                 if bias_cnt[task_id] > err_detect_args.err_patience:  # 小模型相对偏移过大,则更新小模型
    #                     back_up = back_ups[msg_sender]
    #                     model_no = None
    #                     for i in range(len(back_up.models)):
    #                         if task_id in back_up.models[i].member:
    #                             model_no = i
    #                             break
    #                     assert model_no is not None
    #                     new_model = models.models[model_no].get_part(back_up.models[model_no].member)
    #                     new_message = {
    #                         'type': 'model update',
    #                         'update pack': {
    #                             'model_no': model_no,
    #                             'new_model': new_model
    #                         }
    #                     }
    #                     back_up.update(new_message['type'], new_message['update pack'])
    #                     queue_to_workers[task_id].put(new_message)
    #         # 将边缘模型与教师模型对比,判断数据飘移
    #         losses = [criterions[idx](edge_output_plain[idx], targets[idx]) for idx in range(len(related_task_list))]
    #         for list_idx in range(len(related_task_list)):  # 飘移检测和任务发布
    #             task_id = related_task_list[list_idx]
    #             if losses[list_idx] > err_detect_args.acc_threshold:  # 出现单次偏移
    #                 error_cnt[task_id] += 1
    #                 if error_cnt[task_id] > err_detect_args.err_patience:  # 多次偏移,认定为数据飘移
    #                     # 判断组内其他任务拟合是否良好, 决定采用哪种重训练方式
    #                     group_mates = [group_mate for group_mate in range(task_num) if grouping[group_mate] == grouping[task_id]]
    #                     group_avg_error = (np.sum([error_cnt[group_mate] for group_mate in group_mates]) - error_cnt[task_id]) / (len(group_mates) - 1)
    #                     evo_task = None
    #                     if group_avg_error < err_detect_args.regroup_beneath_percent * err_detect_args.err_patience:
    #                         # retrain = TODO: regroup
    #                         evo_args = {}
    #                         evo_task = TrainTask(model=models[group_no], cv_tasks=task_id,
    #                                                  max_epoch=single_prune_args.max_iter,
    #                                                  train_type='reprune', args=retrain_args)
    #                     else:
    #                         # retrain = TODO: reprune
    #                         retrain_args = {}
    #                         evo_task = TrainTask(model=models[group_no], cv_tasks=task_id,
    #                                                  max_epoch=single_prune_args.max_iter,
    #                                                  train_type='reprune', args=retrain_args)
    #                     queue_to_train_manager.put(evo_task)
    #             else:
    #                 error_cnt[task_id] -= 1
    #
    #         # task_id = new_sample_input['task_id']
    #         # bias_grade = new_sample_input['bias_grade']
    #         # biased_sample = new_sample_input['biased_sample']
    #         # old_sub_model = new_sample_input['old_sub_model']
    #         # error_count[task_id] += 1
    #         # # 哪个任务, 偏移水平, 偏移数据, 小模型(要发吗?)
    #         # group_no = grouping[task_id]
    #         # if bias_grade == 0:  # 分组不变, 重算mask
    #         #     retrain_args = {'task_id': task_id, 'biased_sample': biased_sample}
    #         #     retrain_task = TrainTask(model=tree_list[group_no], cv_tasks=task_id,
    #         #                              max_epoch=single_prune_args.max_iter,
    #         #                              train_type='reprune', args=retrain_args)
    #         #     queue_to_train_manager.put(retrain_task)
    #         # # elif ... 其他粒度的重训练策略
    #     if not queue_from_train_manager.empty():
    #         new_ready_signal = queue_from_train_manager.get()
    #         # 哪个任务
    #         task_id = new_ready_signal['task_id']
    #         group_no = grouping[task_id]
    #         assert task_id in models[group_no].member
    #         ingroup_no = models[group_no].member.index(task_id)
    #         sub_model = ModelForrest(model=models[group_no], ingroup_no=ingroup_no)
    #         queue_to_workers[task_id].put(sub_model)
