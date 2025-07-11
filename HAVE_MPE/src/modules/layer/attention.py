import torch as th
import torch.nn as nn
import torch.nn.functional as F

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    b, h, w = matrices.size()
    indices = th.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

# input: X
# key: K=XW_1, query: Q=XW_2, value, V=XW_3
# attention = softmax( QK^{T}/sqrt(d))
# output: attention x V
class SelfAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.tovalues = nn.Linear(self.input_size, self.emb_size * heads, bias = False)

    def forward(self, x):
        b, t, hin = x.size()
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'
        
        h = self.heads 
        e = self.emb_size
        
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))

        dot = th.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        out = th.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return out

# input: X_k, X_q, X_v
# key: K=X_k*W_1, query: Q=X_q*W_2, value, V=X_v*W_3
# attention = softmax( QK^{T}/sqrt(d))
# output: multi-head(attention x V)
class MultiHeadAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, q, k, v, mask=None):
        h = self.heads
        # query shape
        b_q, t_q, e_q = q.size()
        # key shape
        b, t_k, e = k.size()

        # check that key and values have the same batch and embedding dim
        assert b == b_q and e == e_q

        # get keys, queries, values
        keys = self.tokeys(k).view(b, t_k, h, e)
        values = self.tovalues(v).view(b, t_k, h, e)
        queries = self.toqueries(q).view(b, t_q, h, e)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t_k, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t_k, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t_q, e)

        # Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        # - get dot product of queries and keys
        dot = th.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t_q, t_k)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            mask = mask.repeat(h, 1, 1)
            dot = dot.masked_fill(mask == 0, -1e9)

        # dot as row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)
        self.dot = dot # for extract attention
        # apply the self attention to the values
        out = th.bmm(dot, values).reshape(b, h, t_q, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t_q, h * e)

        return self.unifyheads(out)


class SingleHeadAttention(nn.Module):
    def __init__(self, emb, mask=False):
        super().__init__()
        self.emb = emb
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)
        #
        # self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, q, k, v, mask=None):
        # h = self.heads
        # query shape
        b_q, t_q, e_q = q.size()
        # key shape
        b, t_k, e = k.size()

        # check that key and values have the same batch and embedding dim
        assert b == b_q and e == e_q

        # get keys, queries, values
        keys = self.tokeys(k)
        values = self.tovalues(v)
        queries = self.toqueries(q)# b, t_q, e

        # Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        # - get dot product of queries and keys
        dot = th.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b, t_q, t_k)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            # mask = mask.repeat(h, 1, 1)
            dot = dot.masked_fill(mask == 0, -1e9)

        # dot as row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = th.bmm(dot, values)

        return out

class MHA(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, q, k, v, mask=None):
        h = self.heads
        # query shape
        b_q, t_q, e_q = q.size()
        # key shape
        b, t_k, e = k.size()

        # check that key and values have the same batch and embedding dim
        assert b == b_q and e == e_q
        e = e//h
        # get keys, queries, values
        keys = self.tokeys(k).view(b, t_k, h, e)
        values = self.tovalues(v).view(b, t_k, h, e)
        queries = self.toqueries(q).view(b, t_q, h, e)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t_k, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t_k, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t_q, e)

        # Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        # - get dot product of queries and keys
        dot = th.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t_q, t_k)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            mask = mask.repeat(h, 1, 1)
            dot = dot.masked_fill(mask == 0, -1e9)

        # dot as row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = th.bmm(dot, values).reshape(b, h, t_q, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t_q, h * e)

        return self.unifyheads(out)