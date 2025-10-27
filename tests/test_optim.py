"""Optimizer grouping (Muon / AdamW, Qwen-style AdamW), LR curve and S0 / S1 / transition / S2 boundaries."""

import math

import mlx.core as mx
import pytest

from bonsai.model import Bonsai, ModelConfig
from bonsai.optim import (Muon, RunPlan, adamw_peak_lr, build_optimizer, muon_momentum, param_groups, resolve_tc,
                          round_steps, scale_tc, set_lr, set_muon_momentum)


def tiny():
    return Bonsai(ModelConfig.from_preset("tiny", vocab_size=300))


def test_default_groups():
    m = tiny()
    opt, groups = build_optimizer(m)
    assert list(groups) == ["muon", "norm", "embed"]
    assert groups["muon"] == pytest.approx(0.02 * (768 / 128) ** 0.5)
    g = param_groups(m, opt, groups)
    L = m.cfg.n_layer
    assert len(g["muon"]) == 7 * L and all(p.startswith("layers.") and p.endswith("_proj.weight") for p in g["muon"])
    assert len(g["norm"]) == 4 * L + 1 and "norm.weight" in g["norm"]
    assert {"layers.0.self_attn.q_norm.weight", "layers.0.self_attn.k_norm.weight",
            "layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight"} <= set(g["norm"])
    assert g["embed"] == ["embed_tokens.weight"]
    assert isinstance(opt.optimizers[0], Muon)
    assert all(o.weight_decay == 0 for o in opt.optimizers[1:])


def test_qwen_style_adamw_groups():
    m = tiny()
    opt, groups = build_optimizer(m, {"optimizer": "adamw"})
    assert list(groups) == ["decay", "no_decay"]
    g = param_groups(m, opt, groups)
    assert "embed_tokens.weight" in g["decay"] and len(g["decay"]) == 7 * m.cfg.n_layer + 1
    assert all("norm" in p for p in g["no_decay"])
    assert opt.optimizers[0].weight_decay == 0.1 and opt.optimizers[1].weight_decay == 0.0
    assert tuple(opt.optimizers[0].betas) == (0.9, 0.95) and opt.optimizers[0].eps == 1e-8
    assert adamw_peak_lr(576) == pytest.approx(1.5e-3) and adamw_peak_lr(640) == pytest.approx(1.2e-3)
    assert 1.2e-3 < adamw_peak_lr(608) < 1.5e-3 and adamw_peak_lr(256) > 1.5e-3 and adamw_peak_lr(768) < 1.2e-3
    assert groups["decay"] == pytest.approx(adamw_peak_lr(128))


def test_set_lr_and_momentum():
    m = tiny()
    opt, groups = build_optimizer(m)
    set_lr(opt, groups, 0.5)
    for o, base in zip(opt.optimizers, groups.values()):
        assert o.learning_rate.item() == pytest.approx(base * 0.5)
    set_muon_momentum(opt, 0.9)
    assert opt.optimizers[0].state["momentum"].item() == pytest.approx(0.9)
    assert muon_momentum(0) == pytest.approx(0.85) and muon_momentum(150) == pytest.approx(0.90)
    assert muon_momentum(300) == pytest.approx(0.95) and muon_momentum(5000) == pytest.approx(0.95)
    adam, _ = build_optimizer(m, {"optimizer": "adamw"})
    set_muon_momentum(adam, 0.9)  # no Muon: a no-op


def test_scale_tc():
    tc = resolve_tc({}, 640)
    half = scale_tc(tc, 0.15)
    assert half["groups"]["muon"] == pytest.approx(0.15 * tc["groups"]["muon"])
    assert half["norm_lr"] == pytest.approx(0.15e-3)
    opt, groups = build_optimizer(tiny(), half)
    assert groups == half["groups"]


def test_five_day_plan_boundaries():
    total = round_steps(1.41e9, 2 ** 17)
    assert total == 10758
    p = RunPlan.build(total)
    assert (p.s0_end, p.transition, p.decay, p.total) == (538, 7868, 8068, 10758)
    assert p.phase(0) == "s0" and p.phase(537) == "s0" and p.phase(538) == "s1"
    assert p.phase(7867) == "s1" and p.phase(7868) == "transition" and p.phase(8067) == "transition"
    assert p.phase(8068) == "s2" and p.phase(10757) == "s2"
    # token shares match plan section 4 (S0 70M, S1 0.99B, S2 353M)
    tok = 2 ** 17
    assert abs(p.s0_end * tok - 70e6) / 70e6 < 0.02
    assert abs((p.decay - p.s0_end) * tok - 0.99e9) / 0.99e9 < 0.02
    assert abs((p.total - p.decay) * tok - 353e6) / 353e6 < 0.02


def test_wsd_curve():
    p = RunPlan.build(10758)
    assert p.lr(0) == pytest.approx(1 / 300) and p.lr(299) == pytest.approx(1.0)
    assert p.lr(300) == 1.0 and p.lr(537) == 1.0 and p.lr(538) == 1.0  # S0 mostly at peak; hard data switch only
    assert p.lr(8067) == 1.0 and p.lr(8068) == 1.0  # decay starts with S2
    assert p.lr(8069) < 1.0 and p.lr(10757) == pytest.approx(1 / (10758 - 8068))
    vals = [p.lr(s) for s in range(8068, 10758)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert p.lr(9413) == pytest.approx((10758 - 9413) / 2690)


def test_cosine_curve_ends_at_10_percent():
    p = RunPlan.build(1000, schedule="cosine", warmup=100)
    assert p.lr(50) == pytest.approx(51 / 100)
    assert p.lr(100) == pytest.approx(1.0)
    assert p.lr(550) == pytest.approx(0.55)
    assert p.lr(1000) == pytest.approx(0.1)
    assert p.phase(800) == "s2"  # data phases do not depend on the LR schedule


def test_plan_without_s0_or_s2():
    p = RunPlan.build(1000, s0_frac=0)
    assert p.s0_end == 0 and p.phase(0) == "s1" and p.decay == 750 and p.transition == 550
    q = RunPlan.build(1000, s2_frac=0)
    assert q.decay == q.transition == q.total == 1000 and q.phase(999) == "s1" and q.lr(999) == 1.0
    # a transition longer than S1 starts right after S0
    r = RunPlan.build(40, s0_frac=0.25, s2_frac=0.25, transition_steps=100, warmup=5)
    assert (r.s0_end, r.transition, r.decay) == (10, 10, 30)
    with pytest.raises(ValueError):
        RunPlan.build(100, s0_frac=0.8, s2_frac=0.5)
    assert RunPlan(**RunPlan.build(100).to_dict()) == RunPlan.build(100)


def test_warmup_longer_than_run():
    p = RunPlan.build(10, s0_frac=0, s2_frac=0.5, transition_steps=0, warmup=300)
    assert all(p.lr(s) <= (s + 1) / 300 + 1e-12 for s in range(10))
    assert math.isclose(p.lr(9), min(10 / 300, 1 / 5))


SHARDS = {"s0": (80_001_163, "tok/s0"), "s1": (1_200_009_089, "tok/s1"), "s2": (420_000_775, "tok/s2")}


def bench_results(tps):
    from bonsai.model import ModelConfig
    from scripts.bench import ESTIMATE
    out = []
    for cfg, t in tps.items():
        m = Bonsai(ModelConfig.from_preset(cfg, vocab_size=65536))
        out.append({"config": cfg, "micro_batch": 8 if cfg == "bonsai-0.1" else 16, "tok_per_s": t, "peak_mem_gb": 0,
                    "non_embedding_params": m.num_params()["non_embedding"], "vs_estimate": t / ESTIMATE[cfg] - 1})
    return out


def test_bench_decision_rule_reproduces_the_plan():
    """scripts.bench.summarize at the plan's estimated throughputs picks the plan's defaults (section 4)."""
    from scripts.bench import ESTIMATE, summarize

    s = summarize(bench_results(ESTIMATE), shards=SHARDS)
    r50, r96 = s["50h"]["recommended"], s["96h"]["recommended"]
    assert r50["config"] == "bonsai-0.1-mini" and r96["config"] == "bonsai-0.1"
    assert r96["tokens_per_param"] == pytest.approx(17.8, abs=0.05)  # listed as 18 in the plan
    assert r96["tokens"] == pytest.approx(1.41e9) and r50["tokens"] == pytest.approx(0.99e9)
    assert r96["tokens_flag"] == "1.41e9" and r96["micro_batch"] == 8 and "note" not in r96
    assert r96["command"] == "python -m scripts.train --model bonsai-0.1 --micro-batch 8 --tokens 1.41e9 --run bonsai-0.1"
    assert r96["decided_by"].startswith("plan 5-day rule") and r50["decided_by"].startswith("ratio rule")
    assert r96["stages"]["s0"] == 538 * 2 ** 17 and r96["data"]["s2"]["need"] < r96["data"]["s2"]["built"]
    s = summarize(bench_results({**ESTIMATE, "bonsai-0.1": 4500}), shards=SHARDS)
    assert s["96h"]["recommended"]["config"] == "12x640" and "< 4,600" in s["96h"]["recommended"]["decided_by"]
    assert s["per_config"]["bonsai-0.1"]["switch"] is False  # -6% is within 20%
    assert summarize(bench_results({**ESTIMATE, "bonsai-0.1": 3500}), shards=SHARDS)["per_config"]["bonsai-0.1"]["switch"]


def test_bench_follows_the_explicit_5day_rule():
    """4.6k-4.72k tok/s: 14x640 rounds to 17 tokens/param (fails the ratio rule) but the plan keeps it."""
    from scripts.bench import ESTIMATE, summarize
    for tps, tokens in ((4600, "1.35e9"), (4700, "1.38e9"), (4720, "1.38e9")):
        r = summarize(bench_results({**ESTIMATE, "bonsai-0.1": tps}), shards=SHARDS)["96h"]["recommended"]
        assert (r["config"], r["tokens_flag"]) == ("bonsai-0.1", tokens) and ">= 4,600" in r["decided_by"]
        assert r["passes_ratio_rule"] is (tps == 4720)
    # 25% above the estimate and a wider config passes the ratio rule: switch up
    r = summarize(bench_results({**ESTIMATE, "bonsai-0.1": 6000, "16x640": 5600}), shards=SHARDS)["96h"]["recommended"]
    assert r["config"] == "16x640" and r["decided_by"].startswith("switch")
    # nothing passes the ratio rule at 50 h: the plan default, never no recommendation
    r = summarize(bench_results({**ESTIMATE, "bonsai-0.1-mini": 4000, "14x576": 3500}), shards=SHARDS)["50h"]["recommended"]
    assert r["config"] == "bonsai-0.1-mini" and not r["passes_ratio_rule"] and "plan default" in r["decided_by"]


def test_bench_caps_tokens_at_the_shards():
    """S0 (80M) and S2 (420M) run out at ~1.6B tokens, before S1 (1.2B): --tokens is capped with a note."""
    from scripts.bench import ESTIMATE, summarize
    r = summarize(bench_results({**ESTIMATE, "bonsai-0.1": 4500, "12x640": 5600}), shards=SHARDS)["96h"]["recommended"]
    assert r["config"] == "12x640" and r["tokens_measured"] > 1.64e9 and r["tokens_flag"] == "1.60e9"
    assert "S0 would need" in r["note"] and "S2 would need" in r["note"] and "S1" not in r["note"]
    assert all(v["need"] <= v["built"] for v in r["data"].values())


def test_phase_tokens():
    """Per-phase token needs of the 5-day plan (S1 / S2 each get half the transition)."""
    p = RunPlan.build(round_steps(1.41e9, 2 ** 17))
    need = p.phase_tokens(2 ** 17)
    assert need["s0"] == 538 * 2 ** 17 and need["s1"] == (7868 - 538 + 100) * 2 ** 17
    assert need["s2"] == (10758 - 8068 + 100) * 2 ** 17 and sum(need.values()) == p.total * 2 ** 17
    assert RunPlan.build(100, s0_frac=0, s2_frac=0).phase_tokens(10) == {"s0": 0, "s1": 1000, "s2": 0}
