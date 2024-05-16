import multiprocessing

from continuous_grouping.train_task_mgmt.trainQueue import TrainQueue
from continuous_grouping.train_task_mgmt.trainTask import TrainTask
from continuous_grouping.train_task_mgmt.trainer import trainer


def train_manager(max_queue_lvl:int, trainer_num:int, queue_from_main:multiprocessing.Queue, queue_to_main:multiprocessing.Queue):
    # 建立训练任务队列
    train_queue = TrainQueue(max_queue_lvl=max_queue_lvl)
    # 建立子进程及通讯队列
    trainers = []
    queue_to_trainer = []
    queue_from_trainers = multiprocessing.Queue()
    for i in range(trainer_num):
        trainers.append(multiprocessing.Process(target=trainer, args=(queue_to_trainer[i], queue_from_trainers)))
        trainers[i].start()
    while True:
        if not queue_from_trainers.empty():
            new_ready_signal = queue_from_trainers.get()
            # TODO: deal with the ready signal, send it back to main, and then give the trainer another job
        if not queue_from_main.empty():
            new_train_order = queue_from_main.get()
            # TODO: deal with the train order, enqueue it
