import torch
import torch.nn as nn

from init_grouping.model.layerDef import TransformerLayer


class HOINetTransformer(nn.Module):
    def __init__(self,
                 num_layers=2,  # 共用的编码层数(第一层除外)
                 task_dim=8,  # 任务个数(?)
                 model_dim=128,  # (编码后的)任务特征向量维数
                 num_heads=2,  # 注意力头的个数
                 ffn_dim=128,  # 隐藏层维度(以神经元数计)
                 dropout=0.0):
        # Dropout 在训练过程中以一定的概率（通常为 0.5）
        # 随机地将某些神经元的输出置为零，而在测试过程中则
        # 保持所有神经元的输出不变。这样做可以视为在每一次
        # 训练迭代中训练多个不同的子网络，从而减少了神经元
        # 之间的共适应（co-adaptation），降低模型的复
        # 杂度,可以减少过拟合,增强泛化能力,提高健壮性。
        super(HOINetTransformer, self).__init__()
        self.model_dim = model_dim
        # (共用的) transformer 编码模块
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dim_per_head=model_dim,
                dropout=dropout,
                task_dim=task_dim
            ).to('cuda:0')
            for _ in range(num_layers)
        ])
        # (不共用的) 倒数第二层,全连接层,任务相关输出
        self.task_output = [
            nn.Linear(ffn_dim, 1)
            .to('cuda:0')
            for _ in range(task_dim)
        ]
        # (共用的) 最后一层,全连接层,任务无关的最终输出(为什么这样设计?)
        self.final_output = nn.Linear(ffn_dim, 1)
        # 任务编码层:将任务的标识(名称,ID等)转换为一个低维度的向量表示,以便模型能够更好得学习任务之间的相关性.
        # 任务编码层也可能出现在输入层之后,中间层,或输出层之前,这里放在最前端
        self.task_embedding = nn.Embedding(task_dim, model_dim)
        # 由nn.Embedding(vocabulary_size, embedding_dim)定义,
        # 第一个参数表示需要编码的类别数量或词汇表大小,
        # 第二个参数表示编码后的嵌入向量长度

    def forward(self, inputs, index):
        output = self.task_embedding(index)
        # index是一个(size, 27)的每行都相同的数组,所以只取第0行就可以
        test_output = self.task_embedding(index)
        # 调用了两次唯一的任务编码模块,而不是创建了两个该模块,所有模块的创建过程在__init__过程中就已经完成了
        # 两次调用使用了相同的index,返回的张量从引用上并不是同一个变量,但
        # 从数值上是几乎相同的,仅有微小的浮点误差

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, inputs)
            pass_mask = torch.ones(len(inputs[0])).to('cuda:0')  # 测试输出不需要注意力屏蔽
            test_output, test_attention = encoder(test_output[0].unsqueeze(0), pass_mask)
            attentions.append(attention)  # TODO:存它干嘛?
        # TODO:先看训练过程吧
        task_embedding = self.task_embedding(index)[0]
        encoder_output = output  # 返回值之二
        output = self.final_output(output)
        result = torch.squeeze(output, 2)
        return result, attentions, task_embedding, encoder_output





