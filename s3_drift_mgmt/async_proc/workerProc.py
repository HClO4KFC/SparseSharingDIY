import time

import keyboard
from torch.utils.data import DataLoader

from datasets.dataLoader import SensorDataset

global raining


def system_output(outs:list, subset_mapping:list, device_no:int):
    for group, map in outs, subset_mapping:
        assert len(group) == len(map)
        for i in range(len(group)):
            print(f'got output stream of subset {i} from device {device_no}')


def handle_hotkey(worker_id, sensor_dataset:SensorDataset):
    sensor_dataset.raining = not sensor_dataset.raining
    if sensor_dataset.raining:
        print(f"{worker_id} is now raining")
    else:
        print(f"{worker_id} is no longer raining")


def worker(args, dataset_path_pre, cv_subset_args, cv_tasks_args):
    forrest = args['forrest']
    worker_no = args['worker_no']
    sample_interval = args['sample_interval']
    queue_from_main = args['queue_from_main']
    queue_to_main = args['queue_to_main']
    forrest.eval()
    last_time = time.time()
    sensor = SensorDataset(path_pre=dataset_path_pre, cv_subset_args=cv_subset_args,
                           label_id_map={args.cv_tasks_args[i].name:args.cv_tasks_args[i].label_id_map for i in range(len(args.cv_tasks_args))})
    keyboard.add_hotkey(str(worker_no), handle_hotkey, args=[worker_no, sensor])
    while True:
        if not queue_from_main.empty():
            new_message = queue_from_main.get()
            assert 'type' in new_message
            if new_message.type == 'model_update':
                assert 'update pack' in new_message
                forrest.update(new_message['type'], new_message['update pack'])
            print(f"model in device {worker_no} is updated.")
        data = sensor.sense_a_frame()
        outs = forrest(data['input'])
        system_output(outs=outs, subset_mapping=forrest.output_subset_mapping, device_no=worker_no)
        # 以一定周期发送抽样数据回主进程
        curr_time = time.time()
        if curr_time - last_time >= sample_interval:
            new_message = {'type': 'sample frame', 'sender': worker_no, 'data': data}
            print(f'end device no.{worker_no} sending a new frame to main...')
            queue_to_main.put(new_message)
