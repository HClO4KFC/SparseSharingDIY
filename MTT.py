from init_grouping.process.trainEvalTest import get_mtg_model
from init_grouping.process.beamSearch import mtg_beam_search
from continuous_grouping.process.mtlModelGen import get_model
from utils.modelPruner import Pruner
from utils.argParse import get_args, str2list
from continuous_grouping.process.pruneMtl import prune, mtl_train

global testing


if __name__ == '__main__':
    args = get_args()
    testing = True if args.testing == '1' else False
    gpu_id = args.gpu_id
    seed = args.seed
    mtg_ensemble_size = args.mtg_ensemble_size
    mtg_end_num = args.mtg_end_num
    mtg_beam_width = args.beam_width
    save_path_pre = args.save_path_pre
    remain_percent = args.remain_percent
    max_pruning_iter = args.max_pruning_iter
    init_masks = args.init_masks
    need_cut = args.need_cut
    mtg_dataset = 'mimic27'
    step = 1

    print('extract mtg model and data set (re-train and re-parse if not ready to use)')
    # TODO:两两训练得出gain_collection(少训练几轮,用loss斜率判断最终loss会到达哪里)
    parsed_data, mtg_ensemble_model = get_mtg_model(
        testing, save_path_pre, mtg_dataset, gpu_id, step,
        mtg_end_num, mtg_ensemble_size, seed)

    # 根据MTG-net确定初始共享组的划分(小规模:枚举;大规模:波束搜索beam-search)
    print('finish the init_grouping with beam-search method...')
    init_grouping = mtg_beam_search(parsed_data, mtg_ensemble_model, 'cuda:' + gpu_id, mtg_beam_width)

    # TODO: 建立分组稀疏参数共享模型(读adashare, sparseSharing)
    # TODO:用adashare的模型结构,
    model = get_model(grouping=init_grouping)

    # TODO:sparseSharing的学习方式:1)单任务模型结构学习;2)多任务学习
    pruner = Pruner(model=model, prune_names=str2list(need_cut, conj=','), remain_percent=remain_percent, max_pruning_iter=max_pruning_iter)
    prune(model=model, pruner=pruner)
    mtl_train(model=model)

    # TODO: (异步)监控任务表现,若有loss飘升,将该任务编号投入重训练队列
