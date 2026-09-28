"""GPT-style decoder-only transformer.

Uses the same building blocks as current open models (Llama, Qwen, OLMo):
  - RMSNorm instead of LayerNorm
  - rotary position embeddings (RoPE)
  - SwiGLU feed-forward layers
  - optional grouped-query attention (n_kv_head < n_head)
  - optional QK-norm for training stability
  - PyTorch's fused attention kernel (flash attention on supported GPUs)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


def rope_tables(seq_len: int, head_dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)  # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate pairs of channels by a position-dependent angle. x: (B, H, T, head_dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        hidden = 64 * math.ceil(hidden / 64)  # round up for faster GPU kernels
        self.gate_up = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin))
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "vocab_size must be set (it comes from the dataset)"
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = rope_tables(cfg.max_seq_len, cfg.d_model // cfg.n_head, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down the layers that write into the residual stream, so its size
        # stays stable no matter how deep the model is (GPT-2 paper).
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def num_params(self, include_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)

        if targets is None:  # generation: only the last position is needed
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def make_optimizer(self, lr, weight_decay, betas, device_type):
        """AdamW with weight decay on matrices only (not on norms)."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop_token=None):
        """Continue each row of idx. No KV cache: simple, and fast enough for small models."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(context)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits[logits < kth] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                if top_p is not None and top_p < 1.0:
                    sorted_probs, order = probs.sort(dim=-1, descending=True)
                    drop = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
                    sorted_probs[drop] = 0.0
                    probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            if stop_token is not None and bool((next_token == stop_token).all()):
                break
        return idx


def flops_per_token(cfg: ModelConfig, n_params: int) -> float:
    """Training FLOPs per token: 6N for the weights plus the attention term (PaLM paper)."""
    return 6 * n_params + 12 * cfg.n_layer * cfg.d_model * cfg.max_seq_len
