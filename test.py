
import numpy as np

# 创建一个示例数组
arr = np.array([10, 5, 8, 20, 15, 3, 18, 25, 12, 7])
             #   0, 1, 2,  3,  4, 5,  6,  7,  8, 9

# 获取最大的 5 个值的索引
max_indices = np.argpartition(arr, -1)[-1:][0]
print(max_indices)