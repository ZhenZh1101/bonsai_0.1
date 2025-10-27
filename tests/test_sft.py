"""SFT sets (encoding, masks, packing, identity samples, the off-policy build), a tiny SFT run and its exact
resume after a signal (bonsai/sft.py)."""

import json
from argparse import Namespace

import mlx.core as mx
import numpy as np
import pytest

from bonsai import chat
from bonsai import sft as S
from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap


@pytest.fixture(scope="module")
def tok():
    qwen = QwenTokenizer()
    return Tokenizer(qwen, VocabMap.for_tests(qwen, n=3000))


def _conv(i, mode="no_think", tools=False):
    user = {"role": "user", "content": f"Question {i}: what is {i} plus Mississippi?"}
    if tools:
        return {"id": f"p{i}:{mode}", "prompt_id": f"p{i}", "mode": mode, "tools": True, "kind": "math",
                "signal": "soft", "messages": chat.tool_trajectory(user["content"], f"print({i}+1)", str(i + 1),
                                                                   f"Answer: {i + 1}")}
    a = {"role": "assistant", "content": f"It is {i} and a river.", "reasoning_content":
         f"Thinking about {i}." if mode == "think" else ""}
    if mode == "no_think" and i % 2:
        a["think_prefilled"] = True
    return {"id": f"p{i}:{mode}", "prompt_id": f"p{i}", "mode": mode, "tools": False, "kind": "chat",
            "signal": "prefill" if a.get("think_prefilled") else "default", "messages": [user, a]}


def test_to_student_expands_mask_over_split_pieces(tok):
    ids, mask = chat.render(tok.qwen, _conv(3)["messages"])
    s, m, n_oov = S.to_student(tok.vmap, ids, mask)
    assert n_oov > 0 and len(s) > len(ids) and len(s) == len(m)          # " Mississippi" etc. are outside S
    assert tok.decode(s) == tok.qwen.decode(ids)
    sup_t = tok.qwen.decode([t for t, k in zip(ids, mask) if k])
    assert tok.decode([t for t, k in zip(s, m) if k]) == sup_t


def test_encode_chats_bos_and_masks(tok):
    convs = [_conv(1, "think"), _conv(2), _conv(3), _conv(4, tools=True)]
    out = S.encode_chats(tok, convs)
    for (ids, mask, _), c in zip(out, convs):
        assert ids[0] == tok.bos_id and mask[0] == 0 and ids.dtype == np.int32
        sup = tok.decode([t for t, k in zip(ids, mask) if k])
        assert "Question" not in sup and sup.endswith("<|im_end|>")
    assert "<think>\nThinking about 1.\n</think>" in tok.decode([t for t, k in zip(*out[0][:2]) if k])
    assert tok.decode([t for t, k in zip(*out[1][:2]) if k]).startswith("<think>\n\n</think>")   # /no_think style
    assert "<think>" not in tok.decode([t for t, k in zip(*out[2][:2]) if k])               # prefilled: not supervised
    tool_sup = tok.decode([t for t, k in zip(*out[3][:2]) if k])
    assert "<tool_call>" in tool_sup and "Answer: 5" in tool_sup and "<tool_response>" not in tool_sup


def test_write_and_pack_round_trip(tok, tmp_path):
    items = [(i, m) for i, m, _ in S.encode_chats(tok, [_conv(k, "think" if k % 3 else "no_think") for k in range(40)])]
    man = S.write_split(tmp_path, "train", items, tok, seq_len=128)
    S.write_split(tmp_path, "val", items[:4], tok, seq_len=128)
    assert man["dtype"] == np.dtype(tok.vmap.shard_dtype).name and man["vocab_sha"] == tok.vmap.sha()
    assert man["total_docs"] == 40 and man["supervised_tokens"] == sum(int(m.sum()) for _, m in items)
    data = S.SFTData("train", 128, tok.bos_id, tmp_path, tok.vmap.sha())
    x, y = data.rows(0, seed=1)
    assert x.shape[1] == 128 and y.shape == x.shape
    assert (y >= 0).sum() == man["supervised_tokens"]            # every supervised token is a target exactly once
    assert len(x) <= 40 / 2                                      # best-fit packing: several conversations per row
    x2, _ = data.rows(1, seed=1)
    assert not np.array_equal(x, x2) and np.array_equal(data.rows(0, seed=1)[0], x)
    with pytest.raises(SystemExit):
        S.SFTData("train", 128, tok.bos_id, tmp_path, "other-sha")


def test_identity_samples_modes_and_signals():
    ident = pytest.importorskip("bonsai.identity")
    convs = [{"id": f"c{i}", "category": "name", "messages": [
        {"role": "user", "content": "What's your name?"}, {"role": "assistant", "content": "I'm Bonsai-0.1."},
        {"role": "user", "content": "Who made you?"}, {"role": "assistant", "content": "Zhen made me."}]}
        for i in range(200)]
    cfg = {"think_frac": 0.5, "think_soft_frac": 0.3, "no_think_soft_frac": 0.5, "system_prompt_frac": 0.0}
    out = S.identity_samples(convs, 300, cfg)
    assert len(out) == 300 and len({s["id"] for s in out}) == 300
    modes = [s["mode"] for s in out]
    assert 0.35 < modes.count("think") / 300 < 0.65
    for s in out:
        last, user = s["messages"][-1], s["messages"][-2]
        assert last["role"] == "assistant" and user["role"] == "user"
        if s["mode"] == "think":
            text = user["content"][:-len(" /think")] if s["signal"] == "soft" else user["content"]
            category = "name" if len(s["messages"]) == 2 else None   # first exchange carries the category
            assert last["reasoning_content"] == ident.short_reasoning(text, key=s["id"][9:], category=category,
                                                                      reply=last["content"])
            assert s["signal"] in ("soft", "default") and user["content"].endswith("/think") == (s["signal"] == "soft")
        else:
            assert last["reasoning_content"] == ""
            assert last.get("think_prefilled", False) == (s["signal"] == "prefill")
            assert user["content"].endswith("/no_think") == (s["signal"] == "soft")
    assert {len(s["messages"]) for s in out} == {2, 4}                    # both turns of the conversations get cut


def test_build_offpolicy_sft(tok, tmp_path, monkeypatch):
    pytest.importorskip("bonsai.identity")
    import bonsai.common as common
    (tmp_path / "data" / "offpolicy").mkdir(parents=True)
    (tmp_path / "raw" / "filtered").mkdir(parents=True)
    samples = [_conv(k, m) for k in range(150) for m in ("think", "no_think")] + [_conv(999, tools=True)]
    with open(tmp_path / "data" / "offpolicy" / "kept_s0of1.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in samples)
        f.write(json.dumps(_conv(1000))[:50])                  # gen_offpolicy killed mid-write: skipped
    with open(tmp_path / "raw" / "filtered" / "identity_chat.jsonl", "w") as f:
        for i in range(20):
            f.write(json.dumps({"id": f"c{i}", "category": "name", "messages": [
                {"role": "user", "content": f"Who are you, friend {i}?"},
                {"role": "assistant", "content": "I'm Bonsai-0.1, a small model."}]}) + "\n")
    monkeypatch.setattr(common, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(common, "FILTERED_DIR", tmp_path / "raw" / "filtered")
    monkeypatch.setattr(S, "TOK_DIR", tmp_path / "tok")
    cfg = {"offpolicy_glob": "data/offpolicy/kept_*.jsonl", "shard_dir": "offpolicy", "seq_len": 1024,
           "val_fraction": 0.1, "seed": 1, "system_prompt_frac": 0.0,
           "identity": {"pool": "identity_chat", "frac": 0.02, "think_frac": 0.5},
           "identity_sft": {"shard_dir": "identity_sft", "replay_ratio": 3, "val_fraction": 0.05}}
    rep = S.build_offpolicy_sft(cfg, tok)
    assert rep["general_samples"] == 301 and rep["identity_samples"] == round(0.02 / 0.98 * 301)
    train = json.loads((tmp_path / "tok" / "offpolicy" / "train_manifest.json").read_text())
    val = json.loads((tmp_path / "tok" / "offpolicy" / "val_manifest.json").read_text())
    assert train["total_docs"] + val["total_docs"] == 301 + rep["identity_samples"]
    assert 10 < val["total_docs"] < 60 and train["has_mask"] and train["bos_id"] == tok.bos_id
    # both modes of a prompt are in the same split
    d = S.SFTData("val", 1024, tok.bos_id, tmp_path / "tok" / "offpolicy")
    qs = [tok.decode(t) for t, _ in d.convs]
    for q in qs:
        if "Question" in q:
            k = q.split("Question ")[1].split(":")[0]
            assert sum(f"Question {k}:" in x for x in qs) == (1 if k == "999" else 2)
    isft = rep["identity_sft"]
    assert isft["identity"] == 40 and isft["replay"] == 120
    assert (tmp_path / "tok" / "identity_sft" / "train_manifest.json").exists()


def _fake_base(tok, path):
    from bonsai import checkpoint as C
    from bonsai.model import build_model
    from bonsai.optim import resolve_tc
    mx.random.seed(0)
    model = build_model("tiny", vocab_size=tok.vocab_size, seq_len=128)
    C.save_weights(path, model, {"model_config": model.cfg.to_dict(), "vocab_sha": tok.vmap.sha(), "kind": "base",
                                 "optim": resolve_tc({}, model.cfg.d_model), "lr_scale": 1.0})
    return path


def _args(tmp_path, base, data, **kw):
    a = dict(run="t", base_run="t", source="base", base=str(base), data=str(data), out_dir=str(tmp_path / "ckpt"),
             log_dir=str(tmp_path / "logs"), epochs=2, steps=0, lr_frac=1.0, warmup=2, batch_tokens=4 * 128,
             micro_batch=4, clip=1.0, log_every=1, eval_every=0, seed=1, memory_limit_gb=8.0, compile=True,
             final_name="", resume="", save_every_min=1e9, keep=3)
    a.update(kw)
    return Namespace(**a)


def test_tiny_sft_run(tok, tmp_path):
    items = [(i, m) for i, m, _ in S.encode_chats(tok, [_conv(k % 7, "think" if k % 2 else "no_think")
                                                        for k in range(120)])]
    data = tmp_path / "set"
    S.write_split(data, "train", items, tok, 128)
    S.write_split(data, "val", items[:8], tok, 128)
    base = _fake_base(tok, tmp_path / "base.safetensors")
    hist = S.run_sft(_args(tmp_path, base, data, lr_frac=3.0), tok=tok)
    assert [e["epoch"] for e in hist["epochs"]] == [1, 2]
    assert hist["epochs"][-1]["val_loss"] < hist["val"][0]["val_loss"] - 0.5       # it learns the repeated convs
    final = tmp_path / "ckpt" / "sft_final.safetensors"
    assert final.is_symlink() and final.resolve().name == f"sft_epoch{hist['final']['epoch']}.safetensors"
    from bonsai import checkpoint as C
    meta = C.load_meta(final)
    assert meta["kind"] == "sft" and meta["vocab_sha"] == tok.vmap.sha() and "groups" in meta["optim"]
    assert meta["optim"]["groups"]["muon"] == pytest.approx(3.0 * meta["base_optim"]["groups"]["muon"])
    # step mode (identity mini-SFT): fixed number of steps from the SFT model, writes final.safetensors
    h2 = S.run_sft(_args(tmp_path, final, data, steps=5, final_name="final.safetensors", lr_frac=0.5), tok=tok)
    assert (tmp_path / "ckpt" / "final.safetensors").exists() and h2["final"]["val_loss"] <= h2["val"][0]["val_loss"]
    fm = C.load_meta(tmp_path / "ckpt" / "final.safetensors")
    assert fm["kind"] == "identity_sft" and fm["base_optim"] == meta["base_optim"]      # LR from the pretraining peak
    assert fm["optim"]["groups"]["muon"] == pytest.approx(0.5 * meta["base_optim"]["groups"]["muon"])


def _train_log(log_dir, stage):
    return [(r["step"], r["loss"], r["grad_norm"]) for r in map(json.loads, open(log_dir / "metrics.jsonl"))
            if r["kind"] == f"{stage}_train"]


def test_sft_resume_matches_uninterrupted(tok, tmp_path, monkeypatch):
    """SIGHUP mid-epoch 1 and SIGTERM mid-epoch 2 (checkpoint + stop), two --resume auto runs: the same steps,
    losses and weights as one uninterrupted run; the identity mini-SFT (step mode) too."""
    import os
    import signal
    import bonsai.optim as O
    from bonsai import checkpoint as C
    items = [(i, m) for i, m, _ in S.encode_chats(tok, [_conv(k % 7, "think" if k % 2 else "no_think")
                                                        for k in range(120)])]
    data = tmp_path / "set"
    S.write_split(data, "train", items, tok, 128)
    S.write_split(data, "val", items[:8], tok, 128)
    base = _fake_base(tok, tmp_path / "base.safetensors")
    spe = len(S.SFTData("train", 128, tok.bos_id, data).rows(0, 1)[0]) // 4
    a, b = tmp_path / "a", tmp_path / "b"
    ref = S.run_sft(_args(a, base, data, lr_frac=3.0), tok=tok)
    ref_id = S.run_sft(_args(a, base, data, steps=6, final_name="final.safetensors"), tok=tok)

    assert spe >= 4
    kills = {(2 * spe, 3): signal.SIGHUP, (2 * spe, spe + 2): signal.SIGTERM, (6, 2): signal.SIGINT}
    orig = O.linear_warmup_decay

    def lwd(step, total, warmup):           # a signal arrives while step `step` of a `total`-step run runs
        if (total, step) in kills:
            os.kill(os.getpid(), kills.pop((total, step)))
        return orig(step, total, warmup)

    monkeypatch.setattr(O, "linear_warmup_decay", lwd)
    h1 = S.run_sft(_args(b, base, data, lr_frac=3.0), tok=tok)
    assert h1["stopped"] == "signal" and h1["step"] == 4 and C.latest(b / "ckpt" / "sft").name == "step_000004"
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL                 # handlers restored
    with pytest.raises(SystemExit, match="earlier attempt"):                 # a fresh start never mixes attempts
        S.run_sft(_args(b, base, data, lr_frac=3.0), tok=tok)
    with pytest.raises(SystemExit, match="settings changed"):
        S.run_sft(_args(b, base, data, lr_frac=3.0, warmup=3, resume="auto"), tok=tok)
    h2 = S.run_sft(_args(b, base, data, lr_frac=3.0, resume="auto"), tok=tok)
    assert h2["stopped"] == "signal" and h2["step"] == spe + 3 and len(h2["epochs"]) == 1
    h3 = S.run_sft(_args(b, base, data, lr_frac=3.0, resume="auto"), tok=tok)
    assert "stopped" not in h3 and h3["epochs"] == ref["epochs"] and h3["val"] == ref["val"]
    assert _train_log(b / "logs", "sft") == _train_log(a / "logs", "sft")        # step for step
    for k in (1, 2):
        wa, wb = (mx.load(str(d / "ckpt" / f"sft_epoch{k}.safetensors")) for d in (a, b))
        assert all(mx.array_equal(wa[n], wb[n]).item() for n in wa)
    assert S.run_sft(_args(b, base, data, lr_frac=3.0, resume="auto"), tok=tok)["epochs"] == ref["epochs"]  # done
    # step mode: SIGINT during step 2 of 6, resumed
    hi = S.run_sft(_args(b, base, data, steps=6, final_name="final.safetensors"), tok=tok)
    assert hi["stopped"] == "signal" and C.latest(b / "ckpt" / "identity_sft").name == "step_000003"
    hi = S.run_sft(_args(b, base, data, steps=6, final_name="final.safetensors", resume="auto"), tok=tok)
    assert hi["val"] == ref_id["val"] and _train_log(b / "logs", "identity_sft") == _train_log(a / "logs", "identity_sft")
    wa, wb = (mx.load(str(d / "ckpt" / "final.safetensors")) for d in (a, b))
    assert all(mx.array_equal(wa[n], wb[n]).item() for n in wa)
