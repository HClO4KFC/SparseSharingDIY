import multiprocessing

from s3_drift_mgmt.async_proc import trainerProc
from s2_continuous_sharing.train_task_mgmt.trainQueue import TrainQueue


def train_manager(max_queue_lvl:int, trainer_num:int, queue_from_main:multiprocessing.Queue, queue_to_main:multiprocessing.Queue):
    # 建立训练任务队列
    train_queue = TrainQueue(max_queue_lvl=max_queue_lvl)
    # 建立子进程及通讯队列
    trainers = []
    queue_to_trainer = []
    queue_from_trainers = multiprocessing.Queue()
    available = []
    for i in range(trainer_num):
        trainers.append(multiprocessing.Process(target=trainerProc.trainer, args=(queue_to_trainer[i], queue_from_trainers)))
        trainers[i].start()
        available.append(True)
    while True:
        # 若有训练器提交了任务, 将任务反馈转交给main, 然后标记该训练器为空闲
        if not queue_from_trainers.empty():
            new_ready_signal = queue_from_trainers.get()
            trainer = new_ready_signal['trainer_no']
            available[trainer] = True
            queue_to_main.put(new_ready_signal['task'])
        # 若有新的训练任务,则将其入队等待
        elif not queue_from_main.empty():
            new_train_order = queue_from_main.get()
            train_queue.enqueue(new_train_order)
            new_train_order.stage = 'waiting'
        # 若有训练器闲着,则从队列中找一件事给它做
        else:
            for trainer_no in range(len(trainers)):
                if available[trainer_no]:
                    new_train_order = train_queue.get_next_task()
                    if not new_train_order.is_same_as(train_queue.idle_task):
                        available[trainer_no] = False
                        new_train_order.stage = "solving"
                        queue_to_trainer[trainer_no].put(new_train_order)
