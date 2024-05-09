need_re_init = False

if __name__ == '__main__':
    # 两两训练得出gain_collection(少训练几轮,用loss斜率判断最终loss会到达哪里)

    # 用gain_collection训练MTG-net,得到所有分组的增益

    # 根据MTG-net确定初始共享组的划分

    # 建立分组稀疏参数共享模型(读adashare, sparseSharing)

    # (异步)监控任务表现,若有loss飘升,将该任务编号投入重训练队列

    pass
