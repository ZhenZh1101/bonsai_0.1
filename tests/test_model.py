"""Bonsai model: exact parameter counts of the plan's tables (sections 4, 5), HF Qwen3 names, KV-cache equivalence,
one compiled optimizer step with fp32 master weights."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from bonsai.model import Bonsai, ModelConfig, loss_fn, make_caches
from bonsai.optim import Muon, RunPlan, build_optimizer, clip_by_global_norm, set_lr

V = 65536


def counts(preset, vocab=V):
    cfg = ModelConfig.from_preset(preset, vocab_size=vocab)
    shapes = {}
    m = Bonsai(cfg)
    for k, v in tree_flatten(m.parameters()):
        shapes[k] = v.size
    per_layer = sum(v for k, v in shapes.items() if k.startswith("layers.0."))
    n = m.num_params()
    return n, per_layer


def test_param_counts_section5_exact():
    n, per_layer = counts("bonsai-0.1")      # 14 x 640
    assert per_layer == 655_360 + 2 * 327_680 + 655_360 + 3_686_400 + 2 * 128 + 2 * 640 == 5_654_016
    assert n["non_embedding"] == 14 * 5_654_016 + 640 == 79_156_864
    assert n["embedding"] == V * 640
    assert n["total"] == 121_099_904
    assert (round(n["total"] / 1e6, 1), round(n["non_embedding"] / 1e6, 1), round(per_layer / 1e6, 2)) == (121.1, 79.2, 5.65)

    n, per_layer = counts("bonsai-0.1-mini")  # 12 x 576
    assert per_layer == 4_314_496 and n["non_embedding"] == 51_774_528 and n["total"] == 89_523_264
    assert (round(n["total"] / 1e6, 1), round(n["non_embedding"] / 1e6, 1), round(per_layer / 1e6, 2)) == (89.5, 51.8, 4.31)


def test_param_counts_section4_table():
    # the table's totals are sums of rounded parts (14x576: 60.40M + 37.75M = 98.15M is listed as 98.1,
    # 12x640: 67.85M + 41.94M = 109.79M as 109.7)
    for preset, nonembed, total in [("14x576", 60.4, 98.1), ("12x640", 67.8, 109.7), ("16x640", 90.5, 132.4)]:
        n, _ = counts(preset)
        assert round(n["non_embedding"] / 1e6, 1) == nonembed, preset
        assert abs(n["total"] / 1e6 - total) <= 0.1, preset


def test_hf_qwen3_parameter_names():
    m = Bonsai(ModelConfig.from_preset("tiny", vocab_size=512))
    names = {k for k, _ in tree_flatten(m.parameters())}
    layer = {k[len("layers.0."):] for k in names if k.startswith("layers.0.")}
    assert layer == {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                     "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                     "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
                     "mlp.up_proj.weight", "mlp.down_proj.weight"}
    assert {"embed_tokens.weight", "norm.weight"} <= names and not any(k.startswith("lm_head") for k in names)


def test_kv_cache_matches_full_forward():
    mx.random.seed(0)
    cfg = ModelConfig(vocab_size=512, n_layer=2, d_model=128, n_head=2, n_kv_head=1, head_dim=64, mlp_hidden=384,
                      seq_len=64)
    m = Bonsai(cfg, compute_dtype=mx.float32)
    for b in m.layers:  # non-zero output projections so the test exercises attention and the MLP
        b.self_attn.o_proj.weight = mx.random.normal(b.self_attn.o_proj.weight.shape) * 0.02
        b.mlp.down_proj.weight = mx.random.normal(b.mlp.down_proj.weight.shape) * 0.02
    ids = mx.random.randint(0, 512, (1, 20))
    full = m(ids)
    caches = make_caches(m)
    a = m(ids[:, :12], caches)
    b = m(ids[:, 12:15], caches)
    c = [m(ids[:, i:i + 1], caches) for i in range(15, 20)]
    inc = mx.concatenate([a, b] + c, axis=1)
    assert np.allclose(np.array(full), np.array(inc), atol=1e-4)


def test_compiled_step_reduces_loss():
    mx.random.seed(0)
    m = Bonsai(ModelConfig.from_preset("tiny", vocab_size=256, seq_len=32))
    opt, groups = build_optimizer(m)
    assert isinstance(opt.optimizers[0], Muon)
    lg = nn.value_and_grad(m, loss_fn)
    state = [m.state, opt.state]

    def step(x, y):
        loss, g = lg(m, x, y)
        g, norm = clip_by_global_norm(g, 1.0)
        opt.update(m, g)
        return loss, norm

    cstep = mx.compile(step, inputs=state, outputs=state)
    data = mx.random.randint(0, 256, (4, 33))
    plan = RunPlan.build(30, s0_frac=0, s2_frac=0.2, transition_steps=0, warmup=3)
    losses = []
    for i in range(30):
        set_lr(opt, groups, plan.lr(i))
        loss, norm = cstep(data[:, :-1], data[:, 1:])
        mx.eval(loss, norm, m.parameters(), opt.state)
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 1.0, losses
    # fp32 master weights survive bf16 compute; the norm weights are trained too
    assert m.embed_tokens.weight.dtype == mx.float32 and m.layers[0].self_attn.q_proj.weight.dtype == mx.float32
    assert m.layers[0].self_attn.q_norm.weight.dtype == mx.float32
    assert not np.allclose(np.array(m.layers[0].input_layernorm.weight), 1.0)


def test_masked_targets_are_ignored():
    mx.random.seed(1)
    m = Bonsai(ModelConfig.from_preset("tiny", vocab_size=128, seq_len=16), compute_dtype=mx.float32)
    x = mx.random.randint(0, 128, (2, 16))
    y = mx.random.randint(0, 128, (2, 16))
    y_masked = mx.where(mx.arange(16)[None] < 8, -1, y)
    full = loss_fn(m, x, y_masked).item()
    from bonsai.model import token_losses
    ce, valid = token_losses(m, x, y)
    expect = (np.array(ce)[:, 8:].sum() / 16)
    assert abs(full - expect) < 1e-5
