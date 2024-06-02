import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
import numpy as np


# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim  # 30
        self.num_heads = num_heads  # 5
        self.attn_dropout = attn_dropout  # 0
        self.head_dim = embed_dim // num_heads  # 30 // 6 =5
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        self.scaling = self.head_dim ** -0.5  # ** ：乘方（指数）根号dk 收缩因子

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))  # 使得in_proj_weight变得可优化
        self.register_parameter('in_proj_bias', None)
        # in_proj_bias 的意思就是一开始的线性变换的偏置。
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn  # FALSE

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()  # false  # data_ptr():返回tensor第一个元素的地址
        kv_same = key.data_ptr() == value.data_ptr()  # false

        # key, value：500*8*30
        tgt_len, bsz, embed_dim = query.size()  # 50，8，30
        # 以下断言都是为了确认参数合法
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:  # false
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:  # false
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:  # 这里
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling  # 根号dk

        if self.bias_k is not None:  # 不进
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 32*5,50, 30/5
        # contiguous() 返回开辟了一块新的存放q的连续内存，并且改变该值会改变原值
        # head_dim = embed_dim(30) // num_heads(5)
        # view(50, 8*5, 6)  -> 40*50*6
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 32*5,147, 30/5
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 32*5,147, 30/5

        src_len = k.size(1)  # 50

        if self.add_zero_attn:  # 没进
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # 32*5,l1,l2
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:  # 进
            try:
                # attn_weights += attn_mask.unsqueeze(0)  # attn_mask：50*500
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, query.shape[0], 1)
                attn_weights = attn_weights.view(self.num_heads, query.shape[1], query.shape[0], -1).masked_fill(
                    attn_mask.cuda(), -np.inf)
                attn_weights = attn_weights.view(self.num_heads * query.shape[1], query.shape[0], -1)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)  # 32*5,50,30/5 attn_weights: 40*50*500  v:40*500*6
        # bmm: bnm 和 bmp 得到 bnp   -> 40*50*6   具体运算看：https://blog.csdn.net/weixin_45573525/article/details/108143684
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # 拼回去，50，32，30
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)  # 32,5,50,147
        attn_weights = attn_weights.sum(dim=1) / self.num_heads  # 对注意力分数的5个头求均值
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        # 以 q = self.in_proj_q(query)为例
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):  # input: query ,end: 30
        # 以 self._in_proj(query, end=self.embed_dim, **kwargs) 为例
        weight = kwargs.get('weight', self.in_proj_weight)  # 30*30
        bias = kwargs.get('bias', self.in_proj_bias)  # shape=90, size:1
        weight = weight[start:end, :]  # 30*30
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
