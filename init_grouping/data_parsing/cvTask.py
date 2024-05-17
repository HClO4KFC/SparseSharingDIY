from utils.lut import use_which_loss_func


class CvTask():
    # static data structure to store basic information about mtl tasks
    def __init__(self, name:str, no:int, dataset=None):
        self.name = name
        self.no = no
        self.data_set = dataset
        self.loss, self.loss_name = use_which_loss_func(task_name=name)

    def load_dataset(self, dataset):
        self.data_set = dataset
