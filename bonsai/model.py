"""Qwen3-0.6B, scaled down (plan section 5).

Every structural choice of Qwen3 is kept, only the sizes shrink:
  per layer: RMSNorm -> q/k/v projections -> RMSNorm on q and k per head (QK-Norm, before RoPE) -> RoPE
  (theta 1e4, 1024 context) -> GQA causal attention (2 query heads per KV head, head_dim 128, scale 1/sqrt(128))
  -> o_proj residual -> RMSNorm -> SwiGLU (intermediate 3x hidden) residual; final RMSNorm; tied embedding.
  RMSNorm (eps 1e-6) carries a learnable weight; no biases, no dropout, no logit softcap.
Init (not disclosed by Qwen3): linear N(0, 0.02), o_proj and down_proj zero (the residual stream starts as
identity), RMSNorm weights 1, embedding N(0, 0.02).

Parameter names follow Hugging Face Qwen3 without the "model." prefix (embed_tokens.weight,
layers.N.self_attn.q_proj.weight, layers.N.input_layernorm.weight, norm.weight ...), so an export to the Qwen
toolchain is a rename plus the vocabulary mapping.

Precision (plan section 7): fp32 master weights, every matmul in the compute dtype (bf16) with its weight cast
on the fly; mx.fast.rms_norm / rope / scaled_dot_product_attention accumulate in fp32; logits are upcast to
fp32 before the cross-entropy.
"""

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

PRESETS = {
    # name: (n_layer, d_model, n_head, n_kv_head, mlp_hidden, head_dim)
    "bonsai-0.1": (14, 640, 8, 4, 1920, 128),        # 5-day build, 79.2M non-embedding / 121.1M total
    "bonsai-0.1-mini": (12, 576, 6, 3, 1728, 128),   # 3-day build, 51.8M / 89.5M
    "bonsai-0.1-probe": (12, 256, 2, 1, 768, 128),   # S0 ablation proxy on the M2 (plan section 9)
    # other rows of the plan's section 4 table, for the day-0 throughput benchmark
    "14x576": (14, 576, 6, 3, 1728, 128),
    "12x640": (12, 640, 8, 4, 1920, 128),
    "16x640": (16, 640, 8, 4, 1920, 128),
    # smoke tests only
    "tiny": (2, 128, 2, 1, 384, 64),
}


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    seq_len: int = 1024
    n_layer: int = 14
    d_model: int = 640
    n_head: int = 8
    n_kv_head: int = 4
    head_dim: int = 128
    mlp_hidden: int = 1920
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02

    @classmethod
    def from_preset(cls, name, **overrides):
        n_layer, d_model, n_head, n_kv, hidden, head_dim = PRESETS[name]
        return cls(n_layer=n_layer, d_model=d_model, n_head=n_head, n_kv_head=n_kv, mlp_hidden=hidden,
                   head_dim=head_dim, **overrides)

    @property
    def q_dim(self):
        return self.n_head * self.head_dim

    @property
    def kv_dim(self):
        return self.n_kv_head * self.head_dim

    def to_dict(self):
        return asdict(self)


class Linear(nn.Module):
    """Bias-free linear layer with an fp32 master weight, computed in x.dtype."""

    def __init__(self, d_in, d_out, std):
        super().__init__()
        self.weight = mx.random.normal((d_out, d_in)) * std if std > 0 else mx.zeros((d_out, d_in))

    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class RMSNorm(nn.Module):
    """RMSNorm with a learnable weight (init 1); mx.fast.rms_norm accumulates in fp32."""

    def __init__(self, dims, eps):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


class Embedding(nn.Module):
    def __init__(self, vocab, dims, std):
        super().__init__()
        self.weight = mx.random.normal((vocab, dims)) * std

    def __call__(self, idx):
        return self.weight[idx]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head, self.n_kv, self.hd, self.theta = cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.rope_theta
        self.q_proj = Linear(cfg.d_model, cfg.q_dim, cfg.init_std)
        self.k_proj = Linear(cfg.d_model, cfg.kv_dim, cfg.init_std)
        self.v_proj = Linear(cfg.d_model, cfg.kv_dim, cfg.init_std)
        self.o_proj = Linear(cfg.q_dim, cfg.d_model, 0.0)  # zero init
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def __call__(self, x, cache=None, mask=None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, T, self.n_head, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, T, self.n_kv, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv, self.hd).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)
        if mask is None:
            mask = "causal" if T > 1 else None
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.hd), mask=mask)
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = Linear(cfg.d_model, cfg.mlp_hidden, cfg.init_std)
        self.up_proj = Linear(cfg.d_model, cfg.mlp_hidden, cfg.init_std)
        self.down_proj = Linear(cfg.mlp_hidden, cfg.d_model, 0.0)  # zero init

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def __call__(self, x, cache=None, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cache, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Bonsai(nn.Module):
    def __init__(self, cfg: ModelConfig, compute_dtype=mx.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.compute_dtype = compute_dtype
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.d_model, cfg.init_std)
        self.layers = [Block(cfg) for _ in range(cfg.n_layer)]
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = Linear(cfg.d_model, cfg.vocab_size, cfg.init_std)

    def hidden(self, idx, caches=None, mask=None):
        x = self.embed_tokens(idx).astype(self.compute_dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, None if caches is None else caches[i], mask)
        return self.norm(x)

    def logits(self, idx, caches=None, mask=None):
        """fp32 logits (B, T, vocab). `mask`: None (causal), or an additive / boolean SDPA mask (padding)."""
        x = self.hidden(idx, caches, mask)
        head = self.embed_tokens.weight if self.cfg.tie_embeddings else self.lm_head.weight
        return (x @ head.astype(x.dtype).T).astype(mx.float32)

    def __call__(self, idx, caches=None, mask=None):
        return self.logits(idx, caches, mask)

    def num_params(self):
        from mlx.utils import tree_flatten
        flat = dict(tree_flatten(self.parameters()))
        total = sum(v.size for v in flat.values())
        embed = self.embed_tokens.weight.size + (0 if self.cfg.tie_embeddings else self.lm_head.weight.size)
        return {"total": total, "embedding": embed, "non_embedding": total - embed}

    def flops_per_token(self):
        """6 (N_nonembed + N_embed) + 12 L T d_q (plan section 17): the head matmul counts once."""
        c = self.cfg
        n = self.num_params()["non_embedding"] + c.vocab_size * c.d_model
        return 6 * n + 12 * c.n_layer * c.seq_len * c.q_dim


def token_losses(model, inputs, targets):
    """Per-token fp32 cross entropy; targets == -1 are ignored (loss 0)."""
    logits = model(inputs)
    valid = targets >= 0
    ce = nn.losses.cross_entropy(logits, mx.where(valid, targets, 0), reduction="none")
    return ce * valid, valid


def loss_fn(model, inputs, targets):
    ce, valid = token_losses(model, inputs, targets)
    return ce.sum() / mx.maximum(valid.sum(), 1)


class KVCache:
    """Growing per-layer KV cache (B, n_kv_head, T, head_dim); contexts are <= 1024 tokens."""

    step = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update(self, k, v):
        B, H, T, D = k.shape
        if self.keys is None or self.offset + T > self.keys.shape[2]:
            n = ((self.offset + T + self.step - 1) // self.step) * self.step
            new_k, new_v = mx.zeros((B, H, n, D), k.dtype), mx.zeros((B, H, n, D), v.dtype)
            if self.keys is not None:
                new_k[:, :, : self.offset] = self.keys[:, :, : self.offset]
                new_v[:, :, : self.offset] = self.values[:, :, : self.offset]
            self.keys, self.values = new_k, new_v
        self.keys[:, :, self.offset: self.offset + T] = k
        self.values[:, :, self.offset: self.offset + T] = v
        self.offset += T
        return self.keys[:, :, : self.offset], self.values[:, :, : self.offset]


def make_caches(model):
    return [KVCache() for _ in range(model.cfg.n_layer)]


def build_model(cfg_dict_or_preset, vocab_size=None, seq_len=None, compute_dtype=mx.bfloat16):
    """Model from a preset name or a saved config dict (checkpoint meta["model_config"])."""
    if isinstance(cfg_dict_or_preset, str):
        kw = {}
        if vocab_size is not None:
            kw["vocab_size"] = vocab_size
        if seq_len is not None:
            kw["seq_len"] = seq_len
        cfg = ModelConfig.from_preset(cfg_dict_or_preset, **kw)
    else:
        cfg = ModelConfig(**cfg_dict_or_preset)
    model = Bonsai(cfg, compute_dtype)
    mx.eval(model.parameters())
    return model
