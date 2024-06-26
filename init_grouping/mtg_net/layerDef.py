import torch
import torch.optim
import torch.nn as nn


def softmax(input_, t=1.0):
    ex = torch.exp(input_ / t)
    #   sum = torch.sum(ex, axis=1)
    return ex


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0, task_dim=8):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.task_dim = task_dim

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        attn_mask = 1 - attn_mask
        attn_maskv = attn_mask.unsqueeze(2).repeat(1, 1, self.task_dim)
        attn_maskh = attn_mask.unsqueeze(1).repeat(1, self.task_dim, 1)
        if scale:
            attention = attention * scale
        # print(attn_maskh)
        attention[attn_maskh.bool()] = 1e15 * -1
        attention[attn_maskv.bool()] = 1e15 * -1
        # attention[attn_maskv==0] = torch.Tensor(list(-float('inf')))
        attention = torch.nn.functional.softmax(attention / 0.2, dim=2)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=128, num_heads=2, dim_per_head=128, dropout=0.0, task_dim=8):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = dim_per_head
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout, task_dim)
        self.linear_final = nn.Linear(self.dim_per_head * num_heads, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        attn_mask = attn_mask.repeat(num_heads, 1, 1)
        attn_mask = attn_mask.view(batch_size * num_heads, -1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)
        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        # final linear projection
        # print(context[0])
        # print(context.shape)
        output = self.linear_final(context)
        # # output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)
        # output = self.layer_norm(output)
        # print(output.shape)
        # sys.exit(0)
        # output = context
        return output, attention


#  8 7 128
class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=128, ffn_dim=128, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        output = x.transpose(1, 2)
        # output = self.w2(self.relu(self.w1(output)))
        output = self.w2(self.relu(self.w1(output)))
        # output = self.dropout(output.transpose(1, 2))
        output = output.transpose(1, 2)
        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, model_dim=128, task_dim=8, num_heads=2, ffn_dim=128, dim_per_head=128, dropout=0.0):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dim_per_head=dim_per_head,
                                            dropout=dropout, task_dim=task_dim)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        # print('inputs shape',inputs.shape)
        # print('attn_mask', attn_mask.shape)
        # print(attn_mask)
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        # feed forward network
        output = self.feed_forward(context)
        return output, attention
