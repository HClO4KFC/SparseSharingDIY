import multiprocessing


def trainer(trainer_no:int, queue_from_manager:multiprocessing.Queue, queue_to_manager:multiprocessing.Queue):
    while True:
        new_train_task = queue_from_manager.get()
        new_train_task.solve()
        new_train_task.state = 'solved'
        queue_to_manager.put({'task':new_train_task, 'trainer_no':trainer_no})