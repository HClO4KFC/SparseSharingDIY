import os.path

import matplotlib.pyplot as plt
import numpy as np


def washoff_begin_with(lines, pre:str):
    ans = []
    for line in lines:
        if not line.strip().startswith(pre):
            ans.append(line)
    return ans
def open_and_washoff(file_path:str)->list[str]:
    with open(file_path, 'r') as f:
        lines = [line for line in f]
    # lines = washoff_begin_with(lines, 'file_name =')
    lines = washoff_begin_with(lines, '下轮候选: [')
    lines = washoff_begin_with(lines, '预测新一批数据集候选的迁移增益:')
    lines = washoff_begin_with(lines, '下轮选择: [')
    lines = washoff_begin_with(lines, '多任务试训练,标注一批元数据集:')
    lines = washoff_begin_with(lines, '第')
    lines = washoff_begin_with(lines, '迁移增益比较基准制作')
    lines = washoff_begin_with(lines, '元学习训练迭代')
    lines = washoff_begin_with(lines, '开始元学习数据集标注')
    lines = washoff_begin_with(lines, 'finish the init_grouping')
    lines = washoff_begin_with(lines, 'file_name = ')
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
    # for line in lines:
    #     print(f'"{line}"')
    return lines
def draw_grouped_bars(data, i_enum, j_enum, start_color, width, title, y_label, x_label, x_tick_labels, draw_variance=False):
    fig, ax = plt.subplots()

    ind = np.arange(len(i_enum))
    rect = [None for _ in range(len(j_enum))]
    start_color = [col / 255.0 for col in start_color]

    for j in range(len(j_enum)):
        data_j = [data[i][j] for i in range(len(i_enum))]
        rect[j] = ax.bar(ind+j*width, data_j, width, color=tuple(min(col + j * 0.2 * col, 1) for col in start_color))
        # print(tuple(min(col + j * 0.2 * col, 1) for col in start_color))  # 输出配色
    avg = [sum(data[i]) / len(data[i]) for i in range(len(i_enum))]
    mean_plot = ax.plot(ind+(len(j_enum)/2-0.5)*width, avg, marker='o', linestyle='-', color='black', label='average',  markersize=2.5)
    ax.set_xticks(ind + (len(j_enum)/2-0.5)*width)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    # ax.set_ylim(0, 8000)
    ax.set_xticklabels((label for label in x_tick_labels), rotation=0)
    ax.axhline(0, color='black', linewidth=0.5)
    # ax.legend((rect_j for rect_j in rect), (legend for legend in legends))

    if draw_variance:
        std = []
        ax2 = ax.twinx()
        for i in range(len(i_enum)):
            # std_i = np.sqrt(sum((data[i][j] - avg[i])**2 for j in range(len(j_enum))))  # 标准差
            std_i = sum([(data[i][j] - avg[i])**2 for j in range(len(j_enum))])  # 方差
            print(sum([(data[i][j] - avg[i])**2 for j in range(len(j_enum))]))
            std.append(std_i)
        ax2.plot(ind + (len(j_enum) / 2 - 0.5) * width, list(std), marker='x', linestyle='--', color='red', label='variation', markersize=3)
        ax2.set_ylabel('variation of transferring gain')

        # 添加图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)
    else:
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels)

    plt.show()


def main():
    param_name = 'meta_train_iter'
    param_enum = list(range(1, 6, 1))
    repeat_time = 3

    file_path_pre = param_name
    file_path = os.path.join(file_path_pre, param_name+'_test.txt')
    j_enum = list(range(repeat_time))
    i_enum = param_enum

    lines = open_and_washoff(file_path)
    # with open(file_path, 'r') as f:
    #     lines = [line for line in f]
    predicted_gain = [[None for _ in j_enum] for _ in i_enum]
    grouping = [[None for _ in j_enum] for _ in i_enum]
    total_time = [[None for _ in j_enum] for _ in i_enum]
    max_mem = [[None for _ in j_enum] for _ in i_enum]

    i = max(i_enum) + 1
    j = max(j_enum) + 1
    for line in lines:
        if line.startswith(f'枚举参数'):
            i = i_enum.index(int(line.split(', ')[0].split('=')[-1]))
            j = j_enum.index(int(line.split('=')[-1]))
            # print(line, i_enum[i], j_enum[j])
        elif line.startswith('initiated hard grouping'):
            # print(line.split('[')[-1].split(']')[0])
            grouping[i][j] = line.split('[')[-1].split(']')[0]
            predicted_gain[i][j] = float(line.split(' ')[-1])
        elif line.startswith('total time'):
            total_time[i][j] = float(line.split(' ')[-1])
        elif line.startswith('Epoch: ['):
            # print(line.split(' ')[-4:])  # 确定最后一个元素都是mem才行
            curr_mem = line.split(' ')[-1]
            if curr_mem.strip() == '':
                continue
            else:
                curr_mem = int(curr_mem)
            max_mem[i][j] = curr_mem if max_mem[i][j] is None else max(max_mem[i][j], curr_mem)
        elif line.startswith('*****实验结束'):
            # print(line)
            i = max(i_enum) + 1
            j = max(j_enum) + 1
        # Epoch: [1]  [  12/2037]  eta: 0:15:38.029469  lr: 0.000310  loss: 1.1271 (1.0463)  loss_classifier: 0.4207 (0.2596)  loss_box_reg: 0.1119 (0.0721)  loss_objectness: 0.5309 (0.5743)  loss_rpn_box_reg: 0.0634 (0.1402)  time: 0.4632  data: 0.0295  TODO:max mem: 3786

    extra_time = 0
    for extra in range(extra_time):
        file_path = os.path.join(file_path_pre, param_name+f'_test_extra{str(extra)}.txt')
        lines = open_and_washoff(file_path)
        i = max(i_enum) + 1
        j = max(j_enum) + 1
        for line in lines:
            if line.startswith(f'枚举参数'):
                i = i_enum.index(int(line.split(', ')[0].split('=')[-1]))
                if extra == 3:
                    j = j_enum.index(int(line.split('=')[-1]) + 1)
                else:
                    j = j_enum.index(int(line.split('=')[-1]))
                # print(line, i_enum[i], j_enum[j])
            elif line.startswith('initiated hard grouping'):
                # print(line.split('[')[-1].split(']')[0])
                grouping[i][j] = line.split('[')[-1].split(']')[0]
                predicted_gain[i][j] = float(line.split(' ')[-1])
            elif line.startswith('total time'):
                total_time[i][j] = float(line.split(' ')[-1])
            elif line.startswith('Epoch: ['):
                # print(line.split(' ')[-4:])  # 确定最后一个元素都是mem才行
                curr_mem = line.split(' ')[-1]
                if curr_mem.strip() == '':
                    continue
                else:
                    curr_mem = int(curr_mem)
                max_mem[i][j] = curr_mem if max_mem[i][j] is None else max(max_mem[i][j], curr_mem)
            elif line.startswith('*****实验结束'):
                # print(line)
                i = max(i_enum) + 1
                j = max(j_enum) + 1

    for i in range(len(i_enum)):
        for j in range(len(j_enum)):
            print(f'i={i_enum[i]}, j={j_enum[j]}, predicted_gain={predicted_gain[i][j]}, grouping=[{grouping[i][j]}], total_time={total_time[i][j]}, max_mem={max_mem[i][j]}')

    width = 0.2
    title = f'overall time cost with each selected {param_name} value'
    y_label = 'time / s'
    x_label = f'considering values of param {param_name}'
    x_tick_labels = [f'{str(i)}' for i in i_enum]
    start_color = (96, 133, 149)
    draw_grouped_bars(
        data=total_time, i_enum=i_enum, j_enum=j_enum,
        start_color=start_color, width=width, title=title,
        y_label=y_label, x_label=x_label, x_tick_labels=x_tick_labels)

    title = f'overall transferring gain with each selected {param_name} value'
    y_label = 'transferring gain'
    draw_grouped_bars(
        data=predicted_gain, i_enum=i_enum, j_enum=j_enum,
        start_color=start_color, width=width, title=title,
        y_label=y_label, x_label=x_label,
        x_tick_labels=x_tick_labels, draw_variance=False)

    # title = f'maximum GPU memory cost with each {param_name} value'
    # y_label = 'GPU memory cost / MB'
    # draw_grouped_bars(
    #     data=max_mem, i_enum=i_enum, j_enum=j_enum,
    #     start_color=start_color, width=width, title=title,
    #     y_label=y_label, x_label=x_label, x_tick_labels=x_tick_labels)



if __name__ == '__main__':
    main()

