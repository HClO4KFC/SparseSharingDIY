import multiprocessing
import time


def sense_a_frame():
    # TODO: 将rain和/或coarse附加在一部分fine_val后,模拟数据发生偏移的情况
    return []


def system_output(outs:list, subset_mapping:list, device_no:int):
    for group, map in outs, subset_mapping:
        assert len(group) == len(map)
        for i in range(len(group)):
            print(f'got output stream of subset {i} from device {device_no}')
    pass


def worker(args):
    forrest = args['forrest']
    worker_no = args['worker_no']
    sample_interval = args['sample_interval']
    queue_from_main = args['queue_from_main']
    queue_to_main = args['queue_to_main']
    forrest.eval()
    last_time = time.time()
    while True:
        if not queue_from_main.empty():
            new_message = queue_from_main.get()
            if new_message['type'] == 'model update':
                assert 'update pack' in new_message
                forrest.update(new_message['update pack'])
            print(f"model in device {worker_no} is updated.")
        data = sense_a_frame()
        outs = forrest(data)
        system_output(outs=outs, subset_mapping=forrest.output_subset_mapping, device_no=worker_no)
        # 以一定周期发送抽样数据回主进程
        curr_time = time.time()
        if curr_time - last_time >= sample_interval:
            new_message = {'type': 'sample frame', 'sender': worker_no, 'data': data}
            queue_to_main.put(new_message)
