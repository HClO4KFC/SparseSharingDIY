import multiprocessing

import torch.nn

from continuous_grouping.train_task_mgmt.trainTask import TrainTask
from continuous_grouping.process.lut import CvTask


def update_model(new_model_pack):
    # TODO: update the little model of this very task
    pass


def sense_raw_data():
    # TODO: like the behavior of a camera, which take picture with its sensors to further analyse it through the little model
    pass


def parse_raw_data(sensor_data):
    # parse the raw frames into the same layout as model input
    return sensor_data


def accept(access_result):
    # decide whether to accept the bias of this little model
    pass


def assess(out):
    # some forms of analytical methods to show if the model is precise
    pass


def worker(model:torch.nn.Module, max_patience:int, cv_task:CvTask, queue_from_main:multiprocessing.Queue, queue_to_main:multiprocessing.Queue):
    patience = 0
    model.eval()
    while True:
        if not queue_from_main.empty():
            new_model_pack = queue_from_main.get()
            update_model(new_model_pack)
            print("model for task no."+str(cv_task.no)+" is updated.")
        data = parse_raw_data(sense_raw_data())
        out = model(data)
        if not accept(assess(out)):
            patience += 1
            if patience > max_patience:
                retrain_request = TrainTask()
                print("model for task no."+str(cv_task.no)+" requires retraining.")
                queue_to_main.put(retrain_request)
                patience = 0

