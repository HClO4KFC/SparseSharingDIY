import multiprocessing
import omegaconf
import numpy as np
import torch
from torch.utils.data import DataLoader

from s2_continuous_sharing.MultiTaskTrain import multi_task_train
from s3_drift_mgmt.teacherModels import build_teacher_models
from utils.lut import CvTask, select_cv_task
from torchvision import transforms

from s1_init_structure.bootMtgTraining import mtg_active_learning
from s1_init_structure.beamSearch import mtg_beam_search
from s1_init_structure.datasets.dataLoader import MultiDataset
from s1_init_structure.taskRank import mtg_task_rank
from s2_continuous_sharing.rw_lock.RwLock import ReadWriteLock
from s2_continuous_sharing.prunerProc import pruner
from s3_drift_mgmt.async_proc.workerProc import worker
from s3_drift_mgmt.train_task_mgmt.trainTask import TrainTask
from s3_drift_mgmt.async_proc.trainManagerProc import train_manager
from model.mtlModel import SubModel
from poor_old_things.details.mtlDetails import get_models

if __name__ == '__main__':
    # args = get_args()
    args = omegaconf.OmegaConf.load('yamls/default.yaml')
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
    cv_subsets_args = args.cv_subsets_args

    # fix the seed
    np.random.seed(basic_args.seed)
    torch.manual_seed(basic_args.seed)
    torch.cuda.manual_seed_all(basic_args.seed)

    # cal basic info
    task_num = len(cv_tasks_args)

    # 初始化任务描述(内置单任务数据集和加载器)
    task_info_list = [CvTask(no=i, dataset_args=dataset_args,
                             cv_task_arg=cv_tasks_args[i],
                             cv_subsets_args=cv_subsets_args)
                      for i in range(task_num)]

    # 阶段1: 多任务模型构建
    # 准备多任务数据集
    multi_train_dataset = MultiDataset(
        dataset=dataset_args.dataset_name, path_pre=dataset_args.path_pre,
        cv_tasks_args=cv_tasks_args, cv_subsets_args=cv_subsets_args,
        train_val_test='train', transform=transforms.Compose([transforms.ToTensor()]))
    multi_val_dataset = MultiDataset(
        dataset=dataset_args.dataset_name, path_pre=dataset_args.path_pre,
        cv_tasks_args=cv_tasks_args, cv_subsets_args=cv_subsets_args,
        train_val_test='val', transform=transforms.Compose([transforms.ToTensor()]))
    # 元数据集标注+元学习模型mtg-net训练(主动学习策略)
    parsed_data, meta_model = mtg_active_learning(
        multi_train_dataset=multi_train_dataset,
        init_grp_args=init_grp_args,
        mtgnet_args=mtgnet_args,
        dataset_name=dataset_args.dataset_name,
        gpu_id=basic_args.gpu_id,
        backbone=mtl_design_args.backbone,
        out_features=temp_args.out_features,
        task_info_list=task_info_list,
        cv_task_args=cv_tasks_args)

    # 波束搜索确定共享组的划分
    print('finish the init_grouping with beam-search method...')
    grouping = mtg_beam_search(
        task_num=task_num, mtg_model=meta_model,
        device='cuda:' + basic_args.gpu_id, beam_width=beam_search_args.beam_width)

    # pagerank获得任务重要性
    task_ranks, main_tasks = mtg_task_rank(
        task_num=task_num, mtg_model=meta_model, device='cuda:' + basic_args.gpu_id,
        task_rank_args=task_rank_args,grouping=grouping)

    # 初始化参数共享模型(掩膜全通,为硬参数共享)
    models = get_models(grouping=grouping, backbone_name=mtl_design_args.backbone,
                        prune_names=single_prune_args.need_cut, out_features=temp_args.out_features)

    # 阶段2: 硬参数共享过渡到稀疏参数共享

    # 布置进程通信和互斥锁
    dict_lock = ReadWriteLock()
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    for i in range(len(models)):
        shared_dict['model' + str(i)] = None
        shared_dict['mask' + str(i)] = None

    # 异步单任务剪枝
    pruner_args = {
        'dict_lock': dict_lock, 'shared_dict': shared_dict,
        'task_ranks': task_ranks, 'single_prune_args': single_prune_args}
    pruner = multiprocessing.Process(target=pruner, args=pruner_args)
    pruner.start()

    # 多任务平行训练
    multi_task_train(dict_lock=dict_lock, shared_dict=shared_dict,
                     multi_train_args=multi_train_args, cv_tasks_args=cv_tasks_args,
                     models=models, multi_train_dataset=multi_train_dataset,
                     task_info_list=task_info_list, task_ranks=task_ranks)
    pruner.join()

    # 阶段3: 持续演化
    # 启动训练管家进程
    queue_to_train_manager = multiprocessing.Queue()
    queue_from_train_manager = multiprocessing.Queue()
    train_manager = multiprocessing.Process(target=train_manager, args=(
        mgmt_args.max_queue_lvl, mgmt_args.trainer_num, queue_to_train_manager, queue_from_train_manager))
    train_manager.start()

    # 启动工作进程(模拟端设备)
    queue_from_workers = multiprocessing.Queue()
    queue_to_workers = []
    workers = []
    error_count = []
    # 小模型备份
    back_ups = []
    for i in range(len(task_info_list)):
        queue_to_workers.append(multiprocessing.Queue())
        group_no = grouping[i]
        ingroup_no = models[group_no].member.index()
        sub_model = SubModel(model=models, ingroup_no=ingroup_no)
        back_ups.append(sub_model)
        workers.append(multiprocessing.Process(
            target=worker,
            args=(sub_model, mgmt_args.worker_patience, task_info_list[i],
                  multi_train_args.max_iter, queue_to_workers[i], queue_from_workers)))
        workers[i].start()
        error_count.append(0)

    # TODO: 建立教师模型
    teachers = build_teacher_models()

    # 开始持续演化,进程图worker(s)<=(<重分配)(抽样数据>)=>main<=(<重训结果)(重训任务>)=>train_manager<=(<重训结果)(重训任务)=>trainer(s)
    error_cnt = [0 for _ in range(task_num)]
    bias_cnt = [0 for _ in range(task_num)]
    # runtime_dataset_size = err_detect_args.runtime_dataset_size
    # runtime_dataset = [[None for _ in range(len(cv_subsets_args))] for _ in range(runtime_dataset_size)]
    # runtime_dataset_idx = 0

    while True:
        if not queue_from_workers.empty():  # 收到来自端设备的采样数据
            new_sample_input = queue_from_workers.get()
            sensor_id = new_sample_input['sensor_id']
            related_task_list = [i for i in range(task_num) if basic_args.distribution[i] == sensor_id]
            targets = [teachers[task_id](new_sample_input['frame']) for task_id in related_task_list]  # 教师模型推理结果
            model_id = [grouping[rel_task] for rel_task in related_task_list]
            outputs = [model(new_sample_input['frame']) for model in models]  # 大模型推理结果
            outputs = [outputs[model] for model in model_id]
            sub_outputs = [back_ups[rel_task] for rel_task in related_task_list]  # 备份子模型的推演结果
            criterions = [task_info_list[rel_task].loss for rel_task in related_task_list]
            sub_bias = [criterions[idx](sub_outputs[idx], outputs[idx]) for idx in range(len(related_task_list))]
            losses = [criterions[idx](outputs[idx], targets[idx]) for idx in range(len(related_task_list))]
            for list_idx in range(len(related_task_list)):
                task_id = related_task_list[list_idx]
                if sub_bias[list_idx] > err_detect_args.acc_threshold:
                    bias_cnt[task_id] += 1
                    if bias_cnt[task_id] > err_detect_args.err_patience:  # 小模型相对偏移过大,则更换小模型
                        queue_to_workers[task_id].put(SubModel(
                            model=models[model_id[list_idx]],
                            ingroup_no=models[model_id[list_idx]].member.index(task_id)
                        ))

            # runtime_dataset[runtime_dataset_idx] = [None for _ in range(len(cv_subsets_args))]
            # output_names = [cv_tasks_args[rel_task].output for rel_task in related_task_list]
            # output_idxs = []
            # for name in output_names:
            #     index = [idx for idx in range(len(cv_tasks_args)) if cv_tasks_args[idx].name == name]
            #     output_idxs.append(index)
            # for idx in output_idxs:
            #     runtime_dataset[runtime_dataset_idx][idx] = targets[idx]
            for list_idx in range(len(related_task_list)):  # 飘移检测和任务发布
                task_id = related_task_list[list_idx]
                if losses[list_idx] > err_detect_args.acc_threshold:  # 出现单次偏移
                    error_cnt[task_id] += 1
                    if error_cnt[task_id] > err_detect_args.err_patience:  # 多次偏移,认定为数据飘移
                        group_mates = [group_mate for group_mate in range(task_num) if grouping[group_mate] == grouping[task_id]]
                        group_avg_error = (np.sum([error_cnt[group_mate] for group_mate in group_mates]) - error_cnt[task_id]) / (len(group_mates) - 1)
                        evo_task = None
                        if group_avg_error < err_detect_args.regroup_beneath_percent * err_detect_args.err_patience:
                            # retrain = TODO: regroup
                            retrain_args = {}
                            retrain_task = TrainTask(model=models[group_no], cv_tasks=task_id,
                                                     max_epoch=single_prune_args.max_iter,
                                                     train_type='reprune', args=retrain_args)
                        else:
                            # retrain = TODO: reprune
                            retrain_args = {}
                            retrain_task = TrainTask(model=models[group_no], cv_tasks=task_id,
                                                     max_epoch=single_prune_args.max_iter,
                                                     train_type='reprune', args=retrain_args)
                        queue_to_train_manager.put(evo_task)
            # task_id = new_sample_input['task_id']
            # bias_grade = new_sample_input['bias_grade']
            # biased_sample = new_sample_input['biased_sample']
            # old_sub_model = new_sample_input['old_sub_model']
            # error_count[task_id] += 1
            # # 哪个任务, 偏移水平, 偏移数据, 小模型(要发吗?)
            # group_no = grouping[task_id]
            # if bias_grade == 0:  # 分组不变, 重算mask
            #     retrain_args = {'task_id': task_id, 'biased_sample': biased_sample}
            #     retrain_task = TrainTask(model=models[group_no], cv_tasks=task_id,
            #                              max_epoch=single_prune_args.max_iter,
            #                              train_type='reprune', args=retrain_args)
            #     queue_to_train_manager.put(retrain_task)
            # # elif ... 其他粒度的重训练策略
        if not queue_from_train_manager.empty():
            new_ready_signal = queue_from_train_manager.get()
            # 哪个任务
            task_id = new_ready_signal['task_id']
            group_no = grouping[task_id]
            assert task_id in models[group_no].member
            ingroup_no = models[group_no].member.index(task_id)
            sub_model = SubModel(model=models[group_no], ingroup_no=ingroup_no)
            queue_to_workers[task_id].put(sub_model)
