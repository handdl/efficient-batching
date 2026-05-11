import torch
import torch.nn as nn
from dataclasses import dataclass
from einops import rearrange, einsum

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_func = None

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        self.scale = self.head_dim ** 0.5

    def forward(self, x, **kwargs):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.n_heads)
        q = rearrange(q, "b l (h d) -> b l h d", h=self.n_heads)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.n_heads)

        if self.config.use_flash_attn:
            if q.shape[0] == 1 and "cu_seqlens" in kwargs:
                # packing single path — varlen
                q = q.squeeze(0)
                k = k.squeeze(0)
                v = v.squeeze(0)
                cu = kwargs["cu_seqlens"][0].to(torch.int32) # flash_attn_varlen_func needs it :(
                max_seqlen = cu.diff().amax()
                output = flash_attn_varlen_func(q, k, v, cu, cu, max_seqlen, max_seqlen)
                output = rearrange(output, "t h d -> t (h d)")
                return self.out_proj(output).unsqueeze(0)
            else:
                # batched path — regular flash
                output = flash_attn_func(q, k, v, causal=True)
                output = rearrange(output, "b l h d -> b l (h d)")
                return self.out_proj(output)

        attention_mask = kwargs["attention_mask"].to(torch.bool)
        seq = k.size(1)
        causal = torch.tril(torch.ones(seq, seq, device=x.device, dtype=torch.bool))

        if attention_mask.dim() == 2:
            mask = attention_mask[:, None, None, :] & causal
        elif attention_mask.dim() == 3:
            mask = attention_mask[:, None, :, :]

        scores = einsum(q, k, "b l1 h d, b l2 h d -> b h l1 l2") / self.scale
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        attn = einsum(attn, v, "b h l1 l2, b l2 h d -> b l1 h d")
        return self.out_proj(rearrange(attn, "b l1 h d -> b l1 (h d)"))

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim)
        )
        self.attn = SelfAttention(config)
        self.attn_ln = nn.LayerNorm(config.hidden_dim)
        self.ffn_ln = nn.LayerNorm(config.hidden_dim)

    def forward(self, x, **kwargs):
        x += self.attn(self.attn_ln(x), **kwargs)
        x += self.ffn(self.ffn_ln(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.pos_emb = nn.Embedding(config.max_length, config.hidden_dim)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_dim)

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        if "cu_seqlens" in kwargs:
            cu = kwargs["cu_seqlens"][0]
            total_len = input_ids.size(-1)
            pos = torch.cat([torch.arange(cu[j+1] - cu[j], device=input_ids.device) for j in range(len(cu)-1)])
            if pos.size(0) < total_len:
                pos = torch.cat([pos, torch.zeros(total_len - pos.size(0), dtype=torch.long, device=input_ids.device)])
            positions = pos.unsqueeze(0)
        else:
            positions = torch.arange(input_ids.size(-1), device=input_ids.device)

        x = self.emb(input_ids) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.final_ln(x) @ self.emb.weight.T