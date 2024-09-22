import os.path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline

file_path_pre = 'precision'
def dump_0(single_multi):
    task_no = 0
    epoch_s = []
    lr_s = []
    loss_s = []
    mem_s = []
    time_s = []
    sum_time = 0
    file_path = os.path.join(file_path_pre, f'task{str(task_no)}_{single_multi}.txt')
    print(f'file_path={file_path}')
    with open(file_path, 'r') as f:
        for line in f:
            splitted = line.split(' ')
            splitted = [piece for piece in splitted if piece.strip() != '']
            # for i in range(len(splitted)):
            #     print(f'{str(i)}{splitted[i]}')
            epoch = int(splitted[3].split('/')[0])
            lr = float(splitted[7])
            loss = float(splitted[9])
            time = float(splitted[12])+sum_time
            sum_time = time
            mem = int(splitted[-1])
            epoch_s.append(epoch)
            lr_s.append(lr)
            loss_s.append(loss)
            time_s.append(time)
            mem_s.append(mem)
    save_dict={
        'epoch_s':epoch_s,
        'lr_s':lr_s,
        'loss_s':loss_s,
        'time_s':time_s,
        'mem_s':mem_s
    }
    out_path = os.path.join(file_path_pre, f'{single_multi}{str(task_no)}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(save_dict, f)


def dump_n(task_no, single_multi):
    epoch_s = []
    lr_s = []
    loss_s = []
    mem_s = []
    time_s = []
    sum_time = 0
    file_path = os.path.join(file_path_pre, f'task{str(task_no)}_{single_multi}.txt')
    print(f'file_path={file_path}')
    with open(file_path, 'r') as f:
        for line in f:
            splitted = line.split(' ')
            if len(splitted) == 3:
                continue
            splitted = [piece for piece in splitted if piece.strip() != '']
            # Epoch: [0]  [   0/1620]  eta: 0:49:22.952571  lr: 0.000020  loss: 2.6923 (2.6923)  loss_classifier: 1.9600 (1.9600)  loss_box_reg: 0.0001 (0.0001)  loss_objectness: 0.6949 (0.6949)  loss_rpn_box_reg: 0.0374 (0.0374)  time: 1.8290  data: 0.0170  max mem: 1971
            # file_name = dlfip\pascalVOC\VOCdevkit\VOC2012\Annotations\2008_000023.xml
            epoch = int(splitted[3].split('/')[0])
            lr = float(splitted[7])
            loss = float(splitted[9])
            time = float(splitted[24])+sum_time
            sum_time = time
            mem = int(splitted[-1])
            epoch_s.append(epoch)
            lr_s.append(lr)
            loss_s.append(loss)
            time_s.append(time)
            mem_s.append(mem)
    save_dict = {
        'epoch_s': epoch_s,
        'lr_s': lr_s,
        'loss_s': loss_s,
        'time_s': time_s,
        'mem_s': mem_s
    }
    out_path = os.path.join(file_path_pre, f'{single_multi}{str(task_no)}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(save_dict, f)


def draw(task_no):
    dicts = []
    out_path = os.path.join(file_path_pre, f'single{str(task_no)}.pkl')
    with open(out_path, 'rb') as f:
        dict_ = pickle.load(f)
        dicts.append(dict_)
    out_path = os.path.join(file_path_pre, f'multi{str(task_no)}.pkl')
    with open(out_path, 'rb') as f:
        dict_ = pickle.load(f)
        dicts.append(dict_)
    # plt.ylim(2, 2.9)
    # plt.
    i = 0
    for dict_ in dicts:
        i += 1
        x = np.array(dict_['epoch_s'])
        y = np.array(dict_['loss_s'])
        plt.scatter(x[::5], y[::5], s=2, color= 'orange' if i == 1 else 'green')
        def func(x, a, b):
            return a + b * x
        popt, pcov = curve_fit(func, x, y)

        # 绘制拟合曲线
        plt.plot(x, func(x, *popt), color= 'orange' if i == 1 else 'green', label='single-task' if i == 1  else 'multi-task')
    plt.title(f'the trend of loss')
    plt.xlabel('time/s')
    plt.ylabel('loss (linear fit)')

    plt.legend()

    plt.show()


def draw_one_task_mem_cost():

    # 数据
    N = 2
    as_single = [771, 6714]
    as_multi = [373, 4362]
    # static_cost = [373, 771]
    # dynamic_cost = [4362, 6714]

    # 柱形图的宽度
    width = 0.25

    ind = np.arange(N)
    # 绘制柱形图
    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, as_single, width, color=(0.3764705882352941, 0.5215686274509804, 0.5843137254901961))
    rects2 = ax.bar(ind + width, as_multi, width, color=(0.45176470588235296, 0.6258823529411766, 0.7011764705882353))

    # 添加标签、标题以及图例
    ax.set_ylabel('GPU mem cost / MB')
    ax.set_title('GPU mem cost of training a group of 3 tasks\n in single/multi-task ways')
    plt.ylim(0, 10000)
    # ax.set_xticks(ind + width / 2)
    # 调整x轴刻度位置，减小组间的间隙
    ax.set_xticks(ind)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('static cost', 'static & dynamic cost'))
    ax.legend((rects1[0], rects2[0]), ('train as single-tasks', 'train as one group'))

    plt.show()


def draw_space():
    single_mem = []
    multi_mem = []
    single_time = []
    multi_time = []
    for i in range(1, 2):
        with open(os.path.join('precision', f'single{i}.pkl'), "rb") as f:
            dict_ = pickle.load(f)
            single_mem.append(dict_['mem_s'])
            single_time.append(dict_['time_s'])

        with open(os.path.join('precision', f'multi{i}.pkl'), "rb") as f:
            dict_ = pickle.load(f)
            multi_mem.append(dict_['mem_s'])
            multi_time.append(dict_['time_s'])
    single_mem_sum = [sum(column) for column in zip(*single_mem)]
    multi_mem_sum = [sum(column) for column in zip(*multi_mem)]
    single_time_avg = [sum(column)/5 for column in zip(*single_time)]
    multi_time_avg = [sum(column)/5 for column in zip(*multi_time)]
    plt.plot(single_time_avg, single_mem_sum, color='orange', label='train as single task')
    plt.plot(multi_time_avg, multi_mem_sum, color='green', label='train in multi-task group')
    plt.title(f'grouped overall GPU memory cost')
    plt.xlabel('memory cost/MB')
    plt.ylabel('loss (linear fit)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # dump_0('single')
    # dump_n(1, 'single')
    # dump_n(2, 'single')
    # dump_n(3, 'single')
    # dump_n(4, 'single')
    # dump_0('multi')
    # dump_n(1, 'multi')
    # dump_n(2, 'multi')
    # dump_n(3, 'multi')
    # draw_space()
    draw_one_task_mem_cost()
    dump_n(4, 'multi')


