"""Full checkpoint round trip (weights + optimizer state of every group), rotation and setting aside an abandoned
branch, bf16 weights files with meta, bonsai.load.load_model (sources, vocab_sha check) and the HF export."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from bonsai import checkpoint as C
from bonsai.load import load_model, resolve
from bonsai.model import Bonsai, ModelConfig, loss_fn
from bonsai.optim import build_optimizer, set_muon_momentum
from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap


@pytest.fixture(scope="module")
def tok():
    q = QwenTokenizer()
    return Tokenizer(q, VocabMap.for_tests(q, n=300))


def cfg(vocab):
    return ModelConfig.from_preset("tiny", vocab_size=vocab, seq_len=16)


def _train(m, opt, data, steps):
    lg = nn.value_and_grad(m, loss_fn)
    for _ in range(steps):
        _, g = lg(m, data[:, :-1], data[:, 1:])
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)


@pytest.mark.parametrize("optimizer", ["muon", "adamw"])
def test_roundtrip_and_rotation(tmp_path, optimizer):
    mx.random.seed(0)
    data = mx.random.randint(0, 256, (2, 17))
    m = Bonsai(cfg(256))
    opt, _ = build_optimizer(m, {"optimizer": optimizer})
    set_muon_momentum(opt, 0.9)
    _train(m, opt, data, 3)
    for step in (1, 2, 3, 4):
        C.save_full(tmp_path, step, m, opt, {"model_config": m.cfg.to_dict(), "data_state": {"x": step}}, keep=3)
    assert [d.name for d in C._full_dirs(tmp_path)] == ["step_000002", "step_000003", "step_000004"]
    assert C.latest(tmp_path).name == "step_000004" and C.load_meta(C.latest(tmp_path))["data_state"] == {"x": 4}

    # continue A for 2 steps vs. restore into a fresh model B and do the same
    m2 = Bonsai(cfg(256))
    opt2, _ = build_optimizer(m2, {"optimizer": optimizer})
    C.load_weights(m2, C.latest(tmp_path))
    C.load_optimizer(opt2, C.latest(tmp_path))
    if optimizer == "muon":
        assert opt2.optimizers[0].state["momentum"].item() == pytest.approx(0.9)
    _train(m, opt, data, 2)
    _train(m2, opt2, data, 2)
    for (k, a), (_, b) in zip(tree_flatten(m.parameters()), tree_flatten(m2.parameters())):
        assert np.allclose(np.array(a), np.array(b), atol=1e-6), k

    path = C.save_milestone(tmp_path, 5, m, {"model_config": m.cfg.to_dict()})
    assert mx.load(str(path))["embed_tokens.weight"].dtype == mx.bfloat16
    assert C.load_meta(path)["step"] == 5


def test_set_aside_newer(tmp_path):
    """Restarting at step 20 moves the newer checkpoints (abandoned branch) aside: rotation then keeps the new
    branch's checkpoints and latest() returns them."""
    m = Bonsai(cfg(256))
    opt, _ = build_optimizer(m)
    for step in (20, 25, 30):
        C.save_full(tmp_path, step, m, opt, {"model_config": m.cfg.to_dict()}, keep=3)
    moved = C.set_aside_newer(tmp_path, 20)
    assert [p.name for p in moved] == ["step_000025", "step_000030"] and all(p.exists() for p in moved)
    assert moved[0].parent.name.startswith("abandoned_") and C.set_aside_newer(tmp_path, 20) == []
    for step in (22, 24):
        C.save_full(tmp_path, step, m, opt, {"model_config": m.cfg.to_dict()}, keep=3)
    assert [d.name for d in C._full_dirs(tmp_path)] == ["step_000020", "step_000022", "step_000024"]
    assert C.latest(tmp_path).name == "step_000024"


def test_strict_weight_loading(tmp_path):
    m = Bonsai(cfg(256))
    C.save_weights(tmp_path / "w.safetensors", m, {"model_config": m.cfg.to_dict()})
    other = Bonsai(ModelConfig.from_preset("tiny", vocab_size=300, seq_len=16))
    with pytest.raises(ValueError):
        C.load_weights(other, tmp_path / "w.safetensors")
    opt_m, _ = build_optimizer(m)
    C.save_full(tmp_path / "run", 1, m, opt_m, {"model_config": m.cfg.to_dict()})
    opt_a, _ = build_optimizer(m, {"optimizer": "adamw"})
    with pytest.raises(ValueError, match="groups"):
        C.load_optimizer(opt_a, C.latest(tmp_path / "run"))


def test_load_model_sources(tmp_path, tok):
    mx.random.seed(1)
    m = Bonsai(cfg(tok.vocab_size))
    meta = {"model_config": m.cfg.to_dict(), "vocab_sha": tok.vmap.sha()}
    run_dir = tmp_path / "bonsai-test"
    opt, _ = build_optimizer(m)
    C.save_full(run_dir, 7, m, opt, meta)
    assert resolve("bonsai-test", "base", tmp_path).name == "step_000007"  # no base.safetensors yet
    C.save_weights(run_dir / "base.safetensors", m, {**meta, "kind": "base"})
    for source, name in [("base", "base.safetensors"), ("latest", "step_000007")]:
        assert resolve("bonsai-test", source, tmp_path).name == name
    C.save_weights(run_dir / "sft_epoch1.safetensors", m, meta)
    (run_dir / "sft_final.safetensors").symlink_to("sft_epoch1.safetensors")
    assert resolve("bonsai-test", "sft", tmp_path).name == "sft_epoch1.safetensors"
    with pytest.raises(FileNotFoundError, match="onpolicy"):
        resolve("bonsai-test", "onpolicy", tmp_path)

    model, tok2, meta2 = load_model("bonsai-test", "base", tok=tok, ckpt_dir=tmp_path)
    assert tok2 is tok and meta2["kind"] == "base" and model.embed_tokens.weight.dtype == mx.float32
    ids = mx.array([[tok.bos_id, 5, 6, 7]])
    # bf16 storage: logits agree to bf16 precision
    assert np.allclose(np.array(model(ids)), np.array(m(ids)), atol=5e-2)
    model, _, _ = load_model(None, str(run_dir / "step_000007"), tok=tok)
    assert np.array_equal(np.array(model.embed_tokens.weight), np.array(m.embed_tokens.weight))

    C.save_weights(tmp_path / "bad.safetensors", m, {**meta, "vocab_sha": "nope"})
    with pytest.raises(ValueError, match="vocabulary subset"):
        load_model(None, str(tmp_path / "bad.safetensors"), tok=tok)


def export_model(tok, offset):
    """Tiny model; `offset` adds a shared direction to every embedding row, so its final hidden states share one,
    as a trained model's do (a random model's do not, and no linear out-of-S row can be small at all of them)."""
    mx.random.seed(2)
    m = Bonsai(cfg(tok.vocab_size))
    for b in m.layers:
        b.self_attn.o_proj.weight = mx.random.normal(b.self_attn.o_proj.weight.shape) * 0.02
        b.mlp.down_proj.weight = mx.random.normal(b.mlp.down_proj.weight.shape) * 0.02
    v = mx.random.normal((m.cfg.d_model,))
    m.embed_tokens.weight = m.embed_tokens.weight + offset * v / mx.linalg.norm(v)
    return m


def test_export_hf_roundtrip(tmp_path, tok):
    """scripts.export_hf: Qwen3 layout loadable by mlx-lm, logits over S identical to the student's, the shared
    out-of-S row keeps their mass small on held-out raw and chat positions, generation_config suppresses them."""
    import json
    from scripts.export_hf import export
    m = export_model(tok, 0.5)
    rep = export(m, tok, {"vocab_sha": tok.vmap.sha(), "model_config": m.cfg.to_dict()}, tmp_path / "hf", "bonsai-test")
    conf = json.loads((tmp_path / "hf" / "config.json").read_text())
    assert conf["model_type"] == "qwen3" and conf["_name_or_path"] == "bonsai-test" and conf["vocab_size"] == 151936
    assert not conf["tie_word_embeddings"]
    w = mx.load(str(tmp_path / "hf" / "model.safetensors"))
    emb = np.array(w["model.embed_tokens.weight"].astype(mx.float32))
    outside = np.ones(151936, dtype=bool)
    outside[tok.vmap.s2t] = False
    assert emb.shape == (151936, m.cfg.d_model) and not emb[outside].any()
    head = np.array(w["lm_head.weight"].astype(mx.float32))
    out_rows = head[outside]
    assert (out_rows == out_rows[0]).all()
    in_norm = np.linalg.norm(head[~outside], axis=1).max()
    assert np.linalg.norm(out_rows[0]) <= 8 * in_norm * 1.01  # bf16-safe: within a few in-S row norms
    assert "model.layers.0.self_attn.q_norm.weight" in w and "lm_head.weight" in w
    v, oov = rep["verify"], rep["oov"]
    assert v["max_abs_logit_diff_over_S"] < 1e-2 and v["argmax_agree"] == 1.0 and v["out_of_subset_mass_max"] < 1e-4
    assert oov["held_out"]["mass_max"] < 1e-4 and oov["sources"]["chat"]["held_out"]["positions"] > 0
    tc = json.loads((tmp_path / "hf" / "tokenizer_config.json").read_text())
    assert tc["chat_template"].startswith("{{- '<|endoftext|>' }}")
    vocab = json.loads((tmp_path / "hf" / "bonsai_vocab.json").read_text())
    assert vocab["vocab_sha"] == tok.vmap.sha() and vocab["out_of_subset_rows"]["held_out"] == oov["held_out"]
    gen = json.loads((tmp_path / "hf" / "generation_config.json").read_text())
    assert gen["suppress_tokens"] == np.flatnonzero(outside).tolist()
    assert not set(gen["eos_token_id"]) & set(gen["suppress_tokens"])
    assert not list(tmp_path.glob(".tmp_*"))


def test_export_hf_refuses_a_leaking_head(tmp_path, tok):
    """No out-of-S row can be small on the hidden states of a random model: the export fails, nothing written
    (an existing export stays untouched)."""
    from scripts.export_hf import ExportError, export
    m = export_model(tok, 0.0)
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "keep.txt").write_text("previous export")
    with pytest.raises(ExportError, match="nothing written"):
        export(m, tok, {"vocab_sha": tok.vmap.sha(), "model_config": m.cfg.to_dict()}, tmp_path / "hf", "x")
    assert [f.name for f in (tmp_path / "hf").iterdir()] == ["keep.txt"] and not list(tmp_path.glob(".tmp_*"))


def test_out_row_fit_handles_positions_against_the_mean_direction():
    """The old mean-direction method fails when some positions (e.g. BOS / attention-sink states) point against
    the mean hidden state; the fitted row separates them as long as some direction does."""
    from scripts.export_hf import fit_out_row
    rng = np.random.default_rng(0)
    d, n = 32, 4000
    e1, e2 = np.eye(d)[0], np.eye(d)[1]
    H = 3 * e1 + 0.1 * rng.normal(size=(n, d))
    H[:200] = -2 * e1 + 3 * e2 + 0.1 * rng.normal(size=(200, d))  # 5% anti-aligned with the mean
    lse = 5 + 0.1 * rng.random(n)
    c = math.log(86400)
    u = H.mean(0) / np.linalg.norm(H.mean(0))
    assert ((H @ u) <= 0).mean() == 0.05
    w = fit_out_row(H.astype(np.float32), lse.astype(np.float32), c, np.zeros(d, np.float32), radius=20.0)
    a = c + H @ w - lse
    assert np.linalg.norm(w) <= 20.0 * 1.01 and a.max() < math.log(1e-6)
