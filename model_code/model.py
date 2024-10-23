import math
import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass


@dataclass
class ChatGLM2Config():

    hidden_size: int = 4096
    inner_hidden_size: int = 13696
    head_hidden_size: int = 128

    num_multi_query_groups: int = 2
    num_attention_heads: int = 32
    num_layers: int = 28

    vocab_size: int = 65024
    dropout_rate: float = 0.0
    layernorm_epsilon: float = 1e-05
    max_sequence_length: int = 8192


def precompute_for_rotary(dim, length, theta=10000.0):

    dim = dim // 2
    denominator = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    angle = torch.outer(torch.arange(length).float(), denominator)
    rotary = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    rotary_bypass = torch.stack([torch.ones_like(angle), torch.zeros_like(angle)], dim=-1)

    return torch.cat([rotary, rotary_bypass], dim=-2)


def apply_rotary_emb(x, rotary_emb):

    if x.dtype in [torch.float32, torch.float16]:
        x = torch.view_as_complex(x)
        freqs_cis = torch.view_as_complex(rotary_emb)
        return torch.view_as_real(x * freqs_cis).flatten(-2)
    else:
        o_r = x[..., 0] * rotary_emb[..., 0] - x[..., 1] * rotary_emb[..., 1]
        o_i = x[..., 0] * rotary_emb[..., 1] + x[..., 1] * rotary_emb[..., 0]
        return torch.stack([o_r, o_i], dim=-1).flatten(-2)


class RMSNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class ChatGLM2Attention(nn.Module):

    def __init__(self, n_state, n_head, d_head, n_groups, layer_idx, dropout_rate, dtype=None):

        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.n_groups = n_groups
        self.layer_idx = layer_idx
        self.qkv_proj = Linear(n_state, d_head * (n_head + 2 * n_groups), bias=True, dtype=dtype)
        self.o_proj = Linear(d_head * n_head, n_state, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, rotary_emb, attention_mask, kv_cache):

        n_batch, n_seq, _ = x.shape
        d_head, n_head, n_groups = self.d_head, self.n_head, self.n_groups

        fused_qkv = self.qkv_proj(x)
        split_size = [d_head * n_head, d_head * n_groups, d_head * n_groups]
        q, k, v = torch.split(fused_qkv, split_size, dim=-1)

        q = q.view(n_batch, n_seq, n_groups, n_head // n_groups, d_head // 2, 2)
        k = k.view(n_batch, n_seq, n_groups, 1, d_head // 2, 2)
        v = v.view(n_batch, n_seq, n_groups, 1, d_head)

        q = apply_rotary_emb(q, rotary_emb)
        k = apply_rotary_emb(k, rotary_emb)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        kv_cache = (k.detach(), v.detach())

        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 4, 1)
        v = v.permute(0, 2, 3, 1, 4)

        q = q / (math.sqrt(d_head))

        qk = torch.matmul(q, k)
        qk = qk + attention_mask[:, None, None, :, :]

        scores = F.softmax(qk.float(), dim=-1).type_as(x)
        scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        output = output.permute(0, 3, 1, 2, 4).reshape(n_batch, n_seq, -1)
        output = self.o_proj(output)

        return output, kv_cache


class GatedFeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout_rate=0.0, dtype=None):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_in = Linear(dim, hidden_dim * 2, bias=False, dtype=dtype)
        self.w_out = Linear(hidden_dim, dim, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_fn = F.silu

    def forward(self, x):

        h, gate = torch.split(self.w_in(x), self.hidden_dim, dim=-1)

        return self.w_out(self.dropout(self.act_fn(h) * gate))


class ChatGLM2Block(nn.Module):

    def __init__(self, layer_idx, config, dtype=None):

        super().__init__()
        self.layer_idx = layer_idx
        self.attn_ln = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.attn = ChatGLM2Attention(config.hidden_size, config.num_attention_heads, config.head_hidden_size, config.num_multi_query_groups, layer_idx, dropout_rate=config.dropout_rate, dtype=dtype)
        self.ffn_ln = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.ffn = GatedFeedForward(config.hidden_size,  config.inner_hidden_size, config.dropout_rate, dtype=dtype)

    def forward(self, x, rotary_emb, attention_mask, kv_cache):

        h, kv_cache = self.attn(x=self.attn_ln(x), rotary_emb=rotary_emb, attention_mask=attention_mask, kv_cache=kv_cache)
        x = x + h
        h = self.ffn(self.ffn_ln(x))
        output = x + h

        return output, kv_cache


class ChatGLM2Model(nn.Module):

    def __init__(self, config, dtype):

        super().__init__()
        self.config = config
        self.word_embedding = Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, dtype=dtype)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layers = nn.ModuleList([ChatGLM2Block(layer_idx, config, dtype=dtype) for layer_idx in range(config.num_layers)])
        self.final_ln = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        rotary_cache = precompute_for_rotary(config.head_hidden_size, config.max_sequence_length).view(config.max_sequence_length, -1).to(dtype=dtype)
        self.register_buffer("rotary_cache", rotary_cache, persistent=False)

    def prepare_input(self, input_ids, all_kv_cache):

        device = input_ids.device

        n_batch, n_seq_new = input_ids.shape

        if all_kv_cache is not None:
            n_seq_past = all_kv_cache[0][0].shape[1]
            n_seq = n_seq_new + n_seq_past
        else:
            n_seq = n_seq_new

        input_embeddings = self.word_embedding(input_ids)

        attention_mask = torch.ones(n_batch, n_seq, dtype=torch.long, device=device)
        position_ids = torch.cumsum(attention_mask, dim=1)
        seq = torch.arange(n_seq, device=device)
        causal_mask = (seq[:, None] < seq[None, :])
        attention_mask = (causal_mask[None, ...] | ~attention_mask[:, None, :].bool()).float() * -1e10

        attention_mask = attention_mask[:, -n_seq_new:]
        position_ids = position_ids[:, -n_seq_new:]

        rotary_emb = F.embedding(position_ids, self.rotary_cache) .view(n_batch, n_seq_new, 1, 1, self.config.head_hidden_size // 2, 2)

        return (input_embeddings, attention_mask, rotary_emb)

    def forward(self, input_ids, all_kv_cache, labels=None):

        (input_embeddings, attention_mask, rotary_emb) = self.prepare_input(input_ids, all_kv_cache)

        h = self.dropout(input_embeddings)

        current_kv = tuple()

        for i, layer in enumerate(self.layers):

            kv_cache = all_kv_cache[i] if all_kv_cache is not None else None

            h, kv_cache = layer(h, rotary_emb=rotary_emb, attention_mask=attention_mask, kv_cache=kv_cache)

            current_kv += (kv_cache, )

        h = self.final_ln(h)

        output = self.lm_head(h)

        if labels is not None:
            n_classes = self.config.vocab_size
            shift_logits = output[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, n_classes), shift_labels.view(-1))
        else:
            loss = None

        return loss, output, current_kv


class Linear(nn.Linear):

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), None if self.bias is None else self.bias.type_as(x))

    def reset_parameters(self):
        pass


class Embedding(nn.Embedding):

    def reset_parameters(self):
        pass
