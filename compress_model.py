import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.cuda.synchronize()

class MixedModel(nn.Module):
    def __init__(self, batchsize, layers, hidden_dim, ffn_dim, vocab_dim, timesteps, vocab_size):
        super(MixedModel, self).__init__()
        # common params
        self.batchsize = batchsize
        self.timesteps = timesteps
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.vocab_dim = vocab_dim
        self.scale = hidden_dim//vocab_dim
        self.embedding = torch.nn.Embedding(vocab_size, vocab_dim)
        self.lin = nn.Linear(hidden_dim, vocab_size)
        self.W1 = nn.Parameter(torch.zeros(self.layers, dtype=torch.float64), requires_grad=True)
        # self.W2 = nn.Parameter(torch.zeros(self.layers, dtype=torch.float64), requires_grad=True)
        self.count = 0

        # ===== pac =====
        torch.nn.init.normal_(self.embedding.weight, 0, 0.01)
        torch.nn.init.normal_(self.lin.weight, 0, 0.01)
        torch.nn.init.normal_(self.lin.bias, 0, 0.01)
        self.be_ffn = 64
        self.branch = 1
        self.n_group = 2
        self.sgu = True

        mlp = list()
        for i in range(self.layers):
            mlp.append(BELayer(self.branch, self.vocab_dim * (self.n_group ** i), self.be_ffn, batchsize, [i==0, True]))
            mlp.append(LinearLayer(self.timesteps * self.vocab_dim, self.ffn_dim, self.timesteps * self.vocab_dim, 1, self.sgu, [True, True]))

        self.mlp = torch.nn.ModuleList(mlp)
        self.last = list()

    def init_token_order(self, x, module, scale):
        bs, seqlen, vlen = x.shape
        x = x.reshape(bs, seqlen * scale, vlen // scale)        # 这又reshape回去       512, 32, 64
        x_list = list()
        for i in range(seqlen * scale):
            x_list.append(module.forward(x[:, i, :].unsqueeze(1)))
        x = torch.cat(x_list, -1)
        return x

    def forward(self, x):                           # 正常的transformer的forward，计算KQV。训练的时候只训
        x = torch.sigmoid(self.embedding(x))
        bs = x.shape[0]
        # ======= MLP =======
        x = x.reshape(bs, 1, -1)          # 512, 32, 64    512, 8, 256
        last_x = x
        if len(self.last) == 0:
            for i, layer in enumerate(self.mlp):
                if i % 2 == 0:
                    x = self.init_token_order(x, layer, self.timesteps // (pow(self.n_group, i // 2)))
                    self.last.append(x[:, :, self.vocab_dim * (pow(self.n_group, i // 2)):].detach())     # 存放到last里面 len = 4，去掉第一个vocab
                else:
                    # x = layer.forward(x) * torch.sigmoid(self.W1[i // 2]) + last_x * torch.sigmoid(self.W2[i // 2])
                    x = layer.forward(x) * torch.sigmoid(self.W1[i // 2]) + last_x * (1 - torch.sigmoid(self.W1[i // 2]))
                    last_x = x
        else:
            for i, layer in enumerate(self.mlp):
                if i % 2 == 0:          # 这里full_forward只对新的token进行full_forward，然后去拼接之前的
                    new_token = layer.forward(x[:, :, -self.vocab_dim * (pow(self.n_group, i // 2)):])
                    x = torch.cat([self.last[i // 2], new_token], dim=-1)
                    self.last[i // 2] = x[:, :, self.vocab_dim * (pow(self.n_group, i // 2)):].detach()
                else:
                    # x = layer.forward(x) * torch.sigmoid(self.W1[i // 2]) + last_x * torch.sigmoid(self.W2[i // 2])
                    x = layer.forward(x) * torch.sigmoid(self.W1[i // 2]) + last_x * (1 - torch.sigmoid(self.W1[i // 2]))
                    last_x = x
        x = x.reshape(bs, -1, self.hidden_dim)
        final = self.lin(x)
        return final

class dense_baens(nn.Module):
    def __init__(self, N=5, B=4, D1=3, D2=2):
        super(dense_baens, self).__init__()
        self.N = N
        self.B = B
        self.D1 = D1
        self.D2 = D2
        self.U = nn.Parameter(torch.normal(0, 0.01, (N, D1, D2)), requires_grad=True)  # 5 * 3 * 2
        self.bias = nn.Parameter(torch.normal(0, 0.01, (N, B, D2)), requires_grad=True)  # 5 * 4 * 2
    def forward(self, x):
        act = torch.bmm(x, self.U)      # 乘w，加b
        act += self.bias
        return act

# BELayer：
# X reshpae = [bs, branch, vocab_dim]
# X * [bs, vocab_dim, vocab_dim] + [bs, branch, vocab_dim]
# X reshape = [bs, 1, branch * vocab_dim]
class BELayer(torch.nn.Module):
    def __init__(self, branch, vocab_dim, ffn_dim, batch_size, ea=[True, True], trans=False):
        super(BELayer, self).__init__()
        self.branch = branch
        self.vocab_dim = vocab_dim
        self.ffn_dim = ffn_dim
        self.batch_size = batch_size
        # V_map: W=[bs, vocab_dim, vocab_dim], bias=[bs, branch, vocab_dim]
        self.V_map = dense_baens(batch_size, branch, vocab_dim, vocab_dim)

        # https://blog.csdn.net/weixin_39228381/article/details/107939602
        # LayerNorm(
        # normalized_shape:传入一个整数，看作list，传入两个整数，对最后两维归一化
        # eps:归一化加上分母防止除零
        # LayerNorm elementwise_affine=True的话，包含可学习参数Weight和Bias用于仿射变换，即对输入数据归一化到均值0方差1后，乘以weight并加bias
        self.layernorm1 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=ea[0])
        self.layernorm2 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=ea[1])
        self.trans = trans
        self.ln1 = 1

    def forward(self, x):
        x = x.reshape(self.batch_size, self.branch, self.vocab_dim)
        if self.ln1:
            x = self.layernorm1(x)
        skip = x
        x = self.V_map(x)           # 乘W加B，可学习的参数
        x = self.layernorm2(x)      # layernorm
        x = torch.nn.functional.gelu(x)
        x = (skip + x) / 2          # 这里有个残差
        x = x.reshape(self.batch_size, 1, self.branch * self.vocab_dim)     # 合并到一起，维度扩增
        return x

class LinearLayer(torch.nn.Module):

    def __init__(self, hidden_dim, ffn_dim, out_dim, timesteps, if_sgu, ea=[True, True]):
        super(LinearLayer, self).__init__()
        self.if_sgu = if_sgu
        self.U_map = torch.nn.Linear(hidden_dim, 2 * ffn_dim if self.if_sgu else ffn_dim)
        torch.nn.init.normal_(self.U_map.weight, 0, 0.01)
        torch.nn.init.normal_(self.U_map.bias, 0, 0.01)

        self.V_map = torch.nn.Linear(ffn_dim, out_dim)
        torch.nn.init.normal_(self.V_map.weight, 0, 0.01)
        torch.nn.init.normal_(self.V_map.bias, 0, 0.01)

        # self.layernorm1 = torch.nn.LayerNorm(hidden_dim, eps=1e-05, elementwise_affine=ea[0])
        self.layernorm1 = torch.nn.LayerNorm(hidden_dim, eps=1e-05, elementwise_affine=ea[0])
        self.layernorm2 = torch.nn.LayerNorm(out_dim, eps=1e-05, elementwise_affine=ea[1])
        self.ln1 = 1
        self.sgu = SpatialGatingUnit(ffn_dim, timesteps)

    def forward(self, x):
        if self.ln1:
            x = self.layernorm1(x)
        skip = x
        x = self.U_map(x)
        x = torch.nn.functional.gelu(x)
        if self.if_sgu:
            x = self.sgu(x)     # 512, 1, 8192
        x = self.V_map(x)
        x = self.layernorm2(x)
        x = torch.nn.functional.gelu(x)
        x = (skip + x) / 2
        return x

class TinyAttention(nn.Module):
    def __init__(self, d_in, d_out=None, d_attn=64):
        super(TinyAttention, self).__init__()
        self.d_in = d_in
        self.d_out = d_out or d_in
        self.d_attn = d_attn
        self.qkv = nn.Linear(d_in, d_attn * 3)
        self.proj = nn.Linear(d_attn, d_out)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        w = torch.einsum('bnd, bmd->bnm', q, k)
        a = self.softmax(w * torch.rsqrt(torch.tensor(self.d_attn, dtype=torch.float32)))
        x = torch.einsum('bnm, bmd->bnd', a, v)
        out = self.proj(x)
        return out

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len, tiny_attn=False):
        super(SpatialGatingUnit, self).__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.tiny_attn = tiny_attn
        self.tn = TinyAttention(2 * d_ffn, d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)  # dim维度分成两份 2 bs seql dim
        v = self.norm(v)        # 512, 1, 4096
        if self.tiny_attn:
            tn = self.tn(x)
            v = tn + self.spatial_proj(v)
        else:
            v = self.spatial_proj(v)  # 2 bs seql dim * seql seql
        out = u * v  # bs seql dim * bs seql dim     = bs seql dim       门控机制?
        return out