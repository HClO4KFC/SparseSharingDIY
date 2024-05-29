# 用于给有雨的数据输入找到对应的其它数据集所在位置,形成多个map_list
import glob
import os.path
import os
import omegaconf
import winsound

args = omegaconf.OmegaConf.load('D:\\doing\\SparseSharingDIY\\yamls\\default.yaml')
cv_subsets_args = args.cv_subsets_args
subset_num = len(cv_subsets_args)
subset_path_pre = 'D:\\doing\\cvDatasets\\cityscapes'
rain_file_list = glob.glob(os.path.join('D:\\doing\\cvDatasets\\cityscapes\\leftImg8bit_rain', '**', '*leftImg8bit_rain*.png'), recursive=True)

output_path_pre = 'D:\\doing\\cvDatasets\\cityscapes\\leftImg8bit_rain\\gt_map'
output_path = os.path.join(output_path_pre, 'rain.txt')
with open(output_path, "w") as file:
    for line in rain_file_list:
        file.write(line + "\n")
file_lists = [[] for _ in range(subset_num)]
for file_no in range(len(rain_file_list)):
    file = rain_file_list[file_no]
    prefix = '_'.join(file.split('\\')[7].split('_')[0:3])
    print(int(file_no / len(rain_file_list) * 1000) / 10, '% *****', file)
    for subset_no in range(subset_num):
        subset_arg = cv_subsets_args[subset_no]
        if subset_arg.name == 'rain':
            file_lists[subset_no].append(file)
            continue
        subset_file_name_list = glob.glob(os.path.join(subset_path_pre, subset_arg.category, '**', prefix+'_'+subset_arg.set_name+'.'+subset_arg.ext), recursive=True)
        assert len(subset_file_name_list) == 1
        file_lists[subset_no].append(subset_file_name_list[0])
print("saving files...")
for subset_no in range(subset_num):
    output_path = os.path.join(output_path_pre, cv_subsets_args[subset_no].name+'.txt')
    with open(output_path, "w") as file:
        for line in file_lists[subset_no]:
            file.write(line + "\n")
# 发出系统提示音
winsound.MessageBeep()
exit()
