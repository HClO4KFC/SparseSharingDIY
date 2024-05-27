import multiprocessing
import omegaconf
import numpy as np
import torch

from s1_init_structure.datasets.dataLoader import MultiDataset
from s1_init_structure.taskRank import mtg_task_rank
from s3_drift_mgmt.async_proc.workerProc import worker
from model.mtlModel import SubModel
from s1_init_structure.bootMtgTraining import mtg_active_learning
from s1_init_structure.beamSearch import mtg_beam_search
from poor_old_things.details.mtlDetails import get_models
from utils.errReport import CustomError
from s2_continuous_sharing.train_task_mgmt.trainTask import TrainTask
from s3_drift_mgmt.async_proc.trainManagerProc import train_manager
from utils.lut import CvTask
from torchvision import transforms

global testing

if __name__ == '__main__':
    # args = get_args()
    args = omegaconf.OmegaConf.load('yamls/default.yaml')
    basic_args = args.basic_args
    dataset_args = args.dataset_args
    mtgnet_args = args.mtgnet_args
    init_grp_args = args.init_grp_args
    beam_search_args = args.beam_search_args
    mtl_design_args = args.mtl_design_args
    single_prune_args = args.single_prune_args
    multi_train_args = args.multi_train_args
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

    print('extract mtg model and data set (re-train and re-parse if not ready to use)')
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
    parsed_data, model = mtg_active_learning(
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
    init_grouping = mtg_beam_search(
        task_num=task_num, mtg_model=model,
        device='cuda:' + basic_args.gpu_id, beam_width=beam_search_args.beam_width)

    # TODO:pagerank获得任务重要性
    task_rank = mtg_task_rank(
        task_num=task_num, mtg_model=model,
        device='cuda:' + basic_args.gpu_id
    )

    # 初始化参数共享模型(掩膜全通,为硬参数共享)
    models = get_models(grouping=init_grouping, backbone_name=mtl_design_args.backbone,
                        prune_names=single_prune_args.need_cut, out_features=temp_args.out_features)
    # Q: 检查model.train()了吗 A: 在trainer中进行train()和eval()

    # 启动训练管家进程
    queue_to_train_manager = multiprocessing.Queue()
    queue_from_train_manager = multiprocessing.Queue()
    train_manager = multiprocessing.Process(target=train_manager, args=(
        mgmt_args.max_queue_lvl, mgmt_args.trainer_num, queue_to_train_manager, queue_from_train_manager))
    train_manager.start()

    # 多任务预热
    for i in range(len(models)):
        warmup_args = {}
        warmup_task = TrainTask(model=models[i], cv_tasks=[task_info_list[i] for i in models[i].member],
                                max_epoch=temp_args.warmup_iter, train_type="multi_task_warmup", args=warmup_args)
        queue_to_train_manager.put(warmup_task)
    for i in range(len(models)):
        finished_task = queue_from_train_manager.get()
        if finished_task.state != 'solved':
            raise CustomError(finished_task.train_type + " task not solved.")

    # 单任务子模型剪枝
    for i in range(len(task_info_list)):
        prune_args = {'prune_remain_percent': 1-single_prune_args.dec_percent, 'prune_decrease_percent': single_prune_args.dec_percent}
        prune_task = TrainTask(model=models[init_grouping[i] - 1], cv_tasks=[task_info_list[i]],
                               max_epoch=single_prune_args.max_iter, train_type="single_task_prune", args=prune_args)
        queue_to_train_manager.put(prune_task)
    for i in range(len(task_info_list)):
        finished_task = queue_from_train_manager.get()
        if finished_task.state != 'solved':
            raise CustomError(finished_task.train_type + " task not solved.")

    # 多任务同步训练
    for i in range(len(models)):
        mtt_args = {}
        mtl_task = TrainTask(model=models[i], cv_tasks=[task_info_list[i] for i in models[i].member],
                             max_epoch=multi_train_args.max_iter, train_type="multi_task_train", args=mtt_args)
        queue_to_train_manager.put(mtl_task)
    for i in range(len(models)):
        finished_task = queue_from_train_manager.get()
        if finished_task.state != 'solved':
            raise CustomError(finished_task.train_type + " task not solved.")

    # 开始持续演化,进程图worker(s)<---->main<---->train_manager<---->trainer(s)
    queue_from_workers = multiprocessing.Queue()
    queue_to_worker = []
    workers = []
    error_count = []
    for i in range(len(task_info_list)):
        queue_to_worker.append(multiprocessing.Queue())
        group_no = init_grouping[i]
        ingroup_no = models[group_no].member.index()
        sub_model = SubModel(model=models, ingroup_no=ingroup_no)
        workers.append(multiprocessing.Process(
            target=worker,
            args=(sub_model, mgmt_args.worker_patience, task_info_list[i],
                  multi_train_args.max_iter, queue_to_worker, queue_from_workers)))
        workers[i].start()
        error_count.append(0)
    while True:
        if not queue_from_workers.empty():
            new_retrain_request = queue_from_workers.get()  #
            task_id = new_retrain_request['task_id']
            bias_grade = new_retrain_request['bias_grade']
            biased_sample = new_retrain_request['biased_sample']
            old_sub_model = new_retrain_request['old_sub_model']
            error_count[task_id] += 1
            # 哪个任务, 偏移水平, 偏移数据, 小模型(要发吗?)
            group_no = init_grouping[task_id]
            if bias_grade == 0:  # 分组不变, 重算mask
                retrain_args = {'task_id': task_id, 'biased_sample': biased_sample}
                retrain_task = TrainTask(model=models[group_no], cv_tasks=task_id,
                                         max_epoch=single_prune_args.max_iter,
                                         train_type='reprune', args=retrain_args)
                queue_to_train_manager.put(retrain_task)
            # elif ... 其他粒度的重训练策略
        if not queue_from_train_manager.empty():
            new_ready_signal = queue_from_train_manager.get()
            # 哪个任务
            task_id = new_ready_signal['task_id']
            group_no = init_grouping[task_id]
            assert task_id in models[group_no].member
            ingroup_no = models[group_no].member.index(task_id)
            sub_model = SubModel(model=models[group_no], ingroup_no=ingroup_no)
            queue_to_worker[task_id].put(sub_model)
