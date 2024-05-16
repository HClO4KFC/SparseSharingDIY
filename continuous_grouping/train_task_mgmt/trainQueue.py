import torch.nn

from continuous_grouping.train_task_mgmt.trainTask import TrainTask


class TrainQueue():
    def __init__(self, max_queue_lvl):
        self.idle_task = TrainTask(torch.nn.Linear(1, 1), [], 1, 'idle')
        self.train_queues = [[]]  # 工作状态下是二维列表,包含的一维列表个数代表当前活跃的优先级个数,越靠后的一维列表优先级越高;同一个一维列表中的活动处于同一优先级
        self.max_queue_lvl = max_queue_lvl

    def get_next_task(self):
        # 若没有重训练任务,返回默认的闲时任务,空转
        if len(self.train_queues) == 1 and len(self.train_queues[0]) == 0:
            return self.idle_task
        # 从优先级最高的队列上取出最早入队的任务
        for i in range(len(self.train_queues) - 1, -1):
            if len(self.train_queues[i]) == 0:
                    del self.train_queues[i]
            else:
                ans = self.train_queues[i][0]
                # del self.train_queues[i][0]  # trainQueue需要在训练时持续接收训练任务并与队列
                # 中的已有任务合并,因此正在运行的任务仅将其stage设为'solving',但并不删除
                # 直到该训练彻底结束,训练结果更新到小模型为止
                return ans
        # 不论列表是否为空,总应返回训练任务或闲时任务,不应该执行到这里
        assert False

    def look_up(self, train_task):
        for queue in self.train_queues:
            for task in queue:
                if train_task.is_same_as(task):
                    return task
        return None

    def enqueue(self, train_task:TrainTask):
        task = self.look_up(train_task)
        if isinstance(task, TrainTask):
            # 若对应任务已经在队列中,使其忍耐(tolerate),忍不了就将它调入高级队列;
            if not task.tolerate() and task.queue_lvl + 1 < self.max_queue_lvl:
                self.train_queues[task.queue_lvl].remove(task)
                task.queue_lvl += 1
                if task.queue_lvl == len(self.train_queues):
                    self.train_queues.append([])
                self.train_queues[task.queue_lvl].append(task)
        else:
            # 若任务不在队列中, 将其加入队列中
            assert train_task.queue_lvl < self.max_queue_lvl  # 中断恢复的任务应当恢复到其原队列中,这段时间max_queue_lvl应当没有变化
            while len(self.train_queues) - 1 < train_task.queue_lvl:
                self.train_queues.append([])
            self.train_queues[train_task.queue_lvl].append(train_task)

