import multiprocessing

from init_grouping.data_parsing.MtlDataParse import get_mtl_datasets
from continuous_grouping.processes.worker import worker
from init_grouping.process.trainEvalTest import train_mtg_model
from init_grouping.process.beamSearch import mtg_beam_search
from init_grouping.process.details.mtlDetails import get_models
from utils.argParse import get_args, str2list
from utils.lut import init_tasks_info
from continuous_grouping.train_task_mgmt.trainTask import TrainTask
from continuous_grouping.processes.trainManager import train_manager
global testing


if __name__ == '__main__':
    args = get_args()
    testing = True if args.testing == '1' else False
    gpu_id = args.gpu_id
    seed = args.seed
    mtg_dataset = 'mimic27'
    step = 1
    mtg_ensemble_size = args.mtg_ensemble_size
    mtg_end_num = args.mtg_end_num
    mtg_beam_width = args.beam_width
    save_path_pre = args.save_path_pre
    init_masks = args.init_masks

    # build mtl dataset and models
    mtl_dataset_name = args.mtl_dataset_name
    mtl_backbone_name = args.mtl_backbone_name
    mtl_out_features = args.mtl_out_features

    # mtl training management
    max_queue_lvl = args.max_queue_lvl
    trainer_num = args.trainer_num

    # mtl warmup
    warmup_iter = args.warmup_iter

    # single task pruning
    prune_names = str2list(args.need_cut, conj=',')
    prune_remain_percent = args.prune_remain_percent
    max_prune_iter = args.max_prune_iter
    max_mtl_iter = args.max_mtl_iter
    prune_decrease_percent = args.prune_decrease_percent

    # continuous running
    max_worker_patience = args.max_worker_patience
    max_retrain_iter = args.max_retrain_iter

    print('extract mtg model and data set (re-train and re-parse if not ready to use)')
    # TODO:两两训练得出gain_collection(少训练几轮,用loss斜率判断最终loss会到达哪里)
    parsed_data, mtg_ensemble_model = train_mtg_model(
        testing, save_path_pre, mtg_dataset, gpu_id, step,
        mtg_end_num, mtg_ensemble_size, seed)

    # 根据MTG-net确定初始共享组的划分(小规模:枚举;大规模:波束搜索beam-search)
    print('finish the init_grouping with beam-search method...')
    init_grouping = mtg_beam_search(parsed_data, mtg_ensemble_model, 'cuda:' + gpu_id, mtg_beam_width)

    # 获取多任务信息
    task_info_list = init_tasks_info(mtl_dataset_name)
    mtl_datasets = get_mtl_datasets(mtl_dataset_name)
    assert len(task_info_list) == len(mtl_datasets)
    for task_info_i, mtl_dataset_j in task_info_list, mtl_datasets:
        task_info_i.load_dataset(mtl_dataset_j)

    # 建立分组稀疏参数共享模型
    models = get_models(grouping=init_grouping, backbone_name=mtl_backbone_name, out_features=mtl_out_features, prune_names=prune_names)
    # Q: 检查model.train()了吗 A: 在trainer中进行train()和eval()

    # 启动训练管家进程
    queue_to_train_manager = multiprocessing.Queue()
    queue_from_train_manager = multiprocessing.Queue()
    train_manager = multiprocessing.Process(target=train_manager, args=(max_queue_lvl, trainer_num, queue_to_train_manager, queue_from_train_manager))
    train_manager.start()

    # TODO: 多任务预热
    for i in range(len(models)):
        warmup_args = {}
        warmup_task = TrainTask(model=models[i], cv_tasks=[task_info_list[i] for i in models[i].member], max_epoch=warmup_iter, train_type="multi_task_warmup", args=warmup_args)
        queue_to_train_manager.put(warmup_task)
    for i in range(len(models)):
        _ = queue_from_train_manager.get()
        # TODO: get回来点啥

    # TODO: 单任务子模型剪枝
    for i in range(len(task_info_list)):
        prune_args = {'prune_remain_percent': prune_remain_percent, 'prune_decrease_percent': prune_decrease_percent}
        prune_task = TrainTask(model=models[init_grouping[i] - 1], cv_tasks=[task_info_list[i]], max_epoch=max_prune_iter, train_type="single_task_prune", args=prune_args)
        queue_to_train_manager.put(prune_task)
    for i in range(len(task_info_list)):
        _ = queue_from_train_manager.get()
        # TODO: get回来点啥

    # TODO: 多任务同步训练
    for i in range(len(models)):
        mtt_args = {}
        mtl_task = TrainTask(model=models[i], cv_tasks=[task_info_list[i] for i in models[i].member], max_epoch=max_mtl_iter, train_type="multi_task_train", args=mtt_args)
        queue_to_train_manager.put(mtl_task)
    for i in range(len(models)):
        _ = queue_from_train_manager.get()
        # TODO: get回来点啥

    # 开始持续演化,进程图worker(s)<---->main<---->train_manager<---->trainer(s)
    queue_from_workers = multiprocessing.Queue()
    queue_to_worker = []
    workers = []
    for i in range(len(task_info_list)):
        queue_to_worker.append(multiprocessing.Queue())
        sub_model = models[init_grouping[i] - 1].get_sub_model()
        workers.append(multiprocessing.Process(
            target=worker,
            args=(sub_model, max_worker_patience, task_info_list[i],
                  max_retrain_iter, queue_to_worker, queue_from_workers)))
        workers[i].start()
    while True:
        if not queue_from_workers.empty():
            new_retrain_request = queue_from_workers.get()
            # TODO: deal with retrain request, send it to manager
        if not queue_from_train_manager.empty():
            new_ready_signal = queue_from_train_manager.get()
            # TODO: deal with ready signal, send the new model to worker
