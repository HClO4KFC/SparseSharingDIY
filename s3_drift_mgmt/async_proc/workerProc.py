import multiprocessing

import torch.nn

from s2_continuous_sharing.train_task_mgmt.trainTask import TrainTask
from utils.lut import CvTask


def sense_raw_data(cv_task:CvTask):
    pass


def accept(access_result):
    # decide whether to accept the bias of this sub model
    return access_result


def assess(out):
    # some forms of analytical methods to show if the model is precise
    return 1


def worker(model: torch.nn.Module, max_patience: int, cv_task: CvTask, retrain_max_epoch: int,
           queue_from_main: multiprocessing.Queue, queue_to_main: multiprocessing.Queue):
    patience = 0
    model.eval()
    while True:
        if not queue_from_main.empty():
            new_model = queue_from_main.get()
            model = new_model
            print("model for task no." + str(cv_task.no) + " is updated.")
        data = sense_raw_data()
        out = model(data)
        if not accept(assess(out)):
            patience += 1
            if patience > max_patience:
                retrain_args = {}
                retrain_request = TrainTask(model=model, cv_tasks=[cv_task], max_epoch=retrain_max_epoch,
                                            train_type="retrain", args=retrain_args)
                print("model for task no." + str(cv_task.no) + " requires retraining.")
                queue_to_main.put(retrain_request)
                patience = 0
