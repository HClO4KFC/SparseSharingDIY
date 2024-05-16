import multiprocessing


def trainer(queue_from_manager:multiprocessing.Queue, queue_to_manager:multiprocessing.Queue):
    while True:
        new_train_task = queue_from_manager.get()
        new_train_task.solve()
        queue_to_manager.put(new_train_task.get_result())