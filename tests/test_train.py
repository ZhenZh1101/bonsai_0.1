"""End-to-end tiny pretraining run on synthetic shards: S0 (masked, seq 32) -> S1 -> 50/50 transition -> S2 (masked).

A run killed twice (in S0 and in the transition, via the signal path) and resumed with --resume auto reproduces
the uninterrupted run's losses step for step; spike recovery (target = this branch's last checkpoint before the
streak; --skip-steps once per request), the end-of-run base model, resume refusals, the per-phase data check,
the S0 gate line, ckpt_check at the S0 -> S1 switch and the S0 ablation's scaled schedule."""

import json

import mlx.core as mx
import numpy as np
import pytest

from bonsai import checkpoint as C
from bonsai.load import load_model
from bonsai.tokenizer import QwenTokenizer, Tokenizer, VocabMap
from bonsai.train import DONE, STOPPED, Trainer
from scripts.train import build_parser


@pytest.fixture(scope="module")
def tok():
    q = QwenTokenizer()
    return Tokenizer(q, VocabMap.for_tests(q, n=1000))


def write_shards(d, docs, tok, seq_len, masks=None):
    """Synthetic shards in the DESIGN.md 1.2 format (student ids, BOS first, optional masks)."""
    d.mkdir(parents=True, exist_ok=True)
    np.concatenate([np.asarray(x) for x in docs]).astype(tok.vmap.shard_dtype).tofile(d / "shard_00000.bin")
    off = np.zeros(len(docs) + 1, dtype=np.uint64)
    off[1:] = np.cumsum([len(x) for x in docs])
    off.tofile(d / "shard_00000.idx")
    if masks is not None:
        np.concatenate([np.asarray(m) for m in masks]).astype(np.uint8).tofile(d / "shard_00000.mask")
    manifest = {"bos_id": tok.bos_id, "vocab_size": tok.vocab_size, "vocab_sha": tok.vmap.sha(),
                "dtype": np.dtype(tok.vmap.shard_dtype).name, "seq_len": seq_len, "has_mask": masks is not None,
                "total_tokens": int(off[-1]), "total_docs": len(docs),
                "shards": [{"name": "shard_00000", "tokens": int(off[-1]), "docs": len(docs)}], "config": {},
                "sources": {}}
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def s0_docs(rng, n, bos):
    docs, masks = [], []
    for i in range(n):
        if i % 2:  # balanced brackets, full loss
            body, stack = [], []
            while len(body) < int(rng.integers(8, 30)):
                if stack and (rng.random() < 0.5 or len(stack) > 5):
                    body.append(20 + stack.pop())
                else:
                    k = int(rng.integers(0, 4))
                    stack.append(k)
                    body.append(10 + k)
            body += [20 + k for k in reversed(stack)]
            body = body[:32]
            docs.append([bos] + body)
            masks.append([0] + [1] * len(body))
        else:  # derivation: premises ; next: <copy of the 2nd premise>, loss only after the marker
            prem = list(rng.integers(30, 60, int(rng.integers(3, 12))))
            body = prem + [61] + prem[1:3]
            docs.append([bos] + body)
            masks.append([0] * (len(prem) + 2) + [1] * 2)
    return docs, masks


def s1_docs(rng, n, bos):
    out = []
    for _ in range(n):
        a, s, L = int(rng.integers(0, 200)), int(rng.integers(1, 4)), int(rng.integers(5, 64))
        out.append([bos] + [100 + (a + s * k) % 200 for k in range(L)])
    return out


def s2_docs(rng, n, bos):
    docs, masks = [], []
    for _ in range(n):
        p = list(rng.integers(400, 450, int(rng.integers(3, 25))))
        body = p + [501] + p[::-1]
        docs.append([bos] + body)
        masks.append([0] * (len(p) + 2) + [1] * len(p))
    return docs, masks


@pytest.fixture(scope="module")
def shards(tmp_path_factory, tok):
    root = tmp_path_factory.mktemp("shards")
    b = tok.bos_id
    docs, masks = s0_docs(np.random.default_rng(1), 3000, b)
    d = {"s0": write_shards(root / "s0", docs, tok, 32, masks)}
    docs, masks = s0_docs(np.random.default_rng(2), 300, b)
    d["val_s0"] = write_shards(root / "val_s0", docs, tok, 32, masks)
    d["s1"] = write_shards(root / "s1", s1_docs(np.random.default_rng(3), 3000, b), tok, 64)
    d["val"] = write_shards(root / "val", s1_docs(np.random.default_rng(4), 300, b), tok, 64)
    docs, masks = s2_docs(np.random.default_rng(5), 3000, b)
    d["s2"] = write_shards(root / "s2", docs, tok, 64, masks)
    return d


def argv(shards, tmp, run, *extra):
    return ["--model", "tiny", "--seq", "64", "--batch-tokens", "2048", "--micro-batch", "8", "--max-steps", "40",
            "--s0-frac", "0.25", "--s2-frac", "0.25", "--transition-steps", "6", "--warmup", "5",
            "--s0", str(shards["s0"]), "--s1", str(shards["s1"]), "--s2", str(shards["s2"]),
            "--val-dir", str(shards["val"]), "--val-s0-dir", str(shards["val_s0"]), "--log-every", "1",
            "--eval-every", "5", "--eval-tokens", "4096", "--sample-every", "0", "--core-frac", "0",
            "--save-every-min", "1e9", "--milestone-frac", "0.5", "--spike-norm", "100", "--run", run,
            "--ckpt-root", str(tmp / "ckpt"), "--log-root", str(tmp / "logs"), "--memory-limit-gb", "8", *extra]


def run(args, tok, stop_at=None, patch=None):
    tr = Trainer(build_parser().parse_args(args), tok=tok)
    if stop_at is not None:  # behave as if SIGINT arrived during step `stop_at`
        orig = tr.metric

        def metric(**kv):
            orig(**kv)
            if kv.get("kind") == "train" and tr.step == stop_at:
                tr._stop_requested = True

        tr.metric = metric
    if patch:
        patch(tr)
    return tr, tr.train()


def metrics(tmp, run_name, kind):
    rows = [json.loads(l) for l in (tmp / "logs" / run_name / "metrics.jsonl").read_text().splitlines()]
    return [r for r in rows if r["kind"] == kind]


def test_kill_resume_matches_uninterrupted_run(tmp_path, tok, shards):
    tr_a, res = run(argv(shards, tmp_path, "a"), tok)
    assert res == DONE and tr_a.step == 40
    _, res = run(argv(shards, tmp_path, "b"), tok, stop_at=7)                     # killed in S0
    assert res == STOPPED and C.latest(tmp_path / "ckpt" / "b").name == "step_000007"
    _, res = run(argv(shards, tmp_path, "b", "--resume", "auto"), tok, stop_at=27)  # killed in the transition
    assert res == STOPPED and C.latest(tmp_path / "ckpt" / "b").name == "step_000027"
    tr_b, res = run(argv(shards, tmp_path, "b", "--resume", "auto"), tok)
    assert res == DONE and tr_b.step == 40

    la = {m["step"]: m for m in metrics(tmp_path, "a", "train")}
    lb = {m["step"]: m for m in metrics(tmp_path, "b", "train")}
    assert sorted(la) == sorted(lb) == list(range(1, 41))
    diffs = [abs(la[s]["step_loss"] - lb[s]["step_loss"]) for s in range(1, 41)]
    assert max(diffs) < 1e-4, diffs
    assert [la[s]["grad_norm"] for s in (8, 28)] == pytest.approx([lb[s]["grad_norm"] for s in (8, 28)], rel=1e-4)

    # phases and the LR curve as planned: S0 steps 0-9, S1 10-23, transition 24-29, S2 30-39
    phases = [la[s]["phase"] for s in range(1, 41)]
    assert phases == ["s0"] * 10 + ["s1"] * 14 + ["transition"] * 6 + ["s2"] * 10
    assert [la[s]["lr_mult"] for s in range(1, 41)] == pytest.approx([tr_a.plan.lr(s) for s in range(40)])
    assert la[1]["targets"] < la[11]["targets"] == 2048  # S0 derivations are masked, S1 is not
    # the tiny model learns: S1 progressions and the S0 brackets
    assert np.mean([la[s]["step_loss"] for s in range(20, 25)]) < la[11]["step_loss"] - 1.0
    assert la[10]["step_loss"] < la[1]["step_loss"] - 1.0

    for name in ("a", "b"):
        assert len(metrics(tmp_path, name, "phase_switch")) == 1
        s0_vals = metrics(tmp_path, name, "val_s0")
        assert s0_vals and all(m["step"] <= 10 for m in s0_vals)
        vals = metrics(tmp_path, name, "val")
        assert vals[-1]["step"] == 40 and all(np.isfinite(m["val_bpb"]) for m in vals)
    assert metrics(tmp_path, "a", "val")[-1]["val_loss"] == pytest.approx(metrics(tmp_path, "b", "val")[-1]["val_loss"],
                                                                          abs=1e-4)

    # end of pretraining: bf16 base model with meta, loadable through bonsai.load
    base = tmp_path / "ckpt" / "a" / "base.safetensors"
    assert mx.load(str(base))["embed_tokens.weight"].dtype == mx.bfloat16
    model, _, meta = load_model("a", "base", tok=tok, ckpt_dir=tmp_path / "ckpt")
    assert meta["vocab_sha"] == tok.vmap.sha() and meta["kind"] == "base" and meta["step"] == 40
    assert (tmp_path / "logs" / "a" / "pretrain_done.json").exists()
    assert len(list((tmp_path / "ckpt" / "a" / "milestones").glob("*.bf16.safetensors"))) >= 2
    full = C.load_meta(C.latest(tmp_path / "ckpt" / "a"))
    assert set(full["data_state"]) == {"s0", "s1", "s2"} and full["plan"]["s0_end"] == 10

    # resuming with another plan after its boundaries passed, or with another vocabulary, is refused
    with pytest.raises(SystemExit, match="refusing to change the run plan"):
        Trainer(build_parser().parse_args(argv(shards, tmp_path, "a", "--resume", "auto", "--max-steps", "50")), tok=tok)
    q = tok.qwen
    other = Tokenizer(q, VocabMap.for_tests(q, n=900))
    with pytest.raises(SystemExit, match="vocabulary subset"):
        Trainer(build_parser().parse_args(argv(shards, tmp_path, "c")), tok=other)


def test_spike_recovery(tmp_path, tok, shards):
    fired = []

    def nan_once(tr):
        orig = tr._apply

        def apply(grads):
            norm = orig(grads)
            if tr.step == 12 and not fired:
                fired.append(tr.step)
                return mx.array(float("nan"))
            return norm

        tr._apply = apply

    args = argv(shards, tmp_path, "r", "--save-every", "5", "--recover-skip-steps", "3", "--max-steps", "24",
                "--milestone-frac", "0")
    tr, res = run(args, tok, patch=nan_once)
    assert res == DONE and fired == [12] and tr.step == 24
    rec = metrics(tmp_path, "r", "recover")
    assert len(rec) == 1 and rec[0]["step"] == 10 and rec[0]["lr_scale"] == pytest.approx(0.7)
    assert tr.stats["recoveries"] == 1 and tr.stats["skipped_steps"] == 3
    meta = C.load_meta(C.latest(tmp_path / "ckpt" / "r"))
    assert meta["lr_scale"] == pytest.approx(0.7)
    lr = {m["step"]: m["lr_mult"] for m in metrics(tmp_path, "r", "train")}
    assert lr[24] == pytest.approx(0.7 * tr.plan.lr(23))

    # no recovery left: stop with exit code 2
    fired.clear()
    args = argv(shards, tmp_path, "r2", "--save-every", "5", "--auto-recover", "0", "--max-steps", "24")
    with pytest.raises(SystemExit) as e:
        run(args, tok, patch=nan_once)
    assert e.value.code == 2


def nan_at(*steps):
    """Patch: a NaN gradient norm once at each of these (0-based) steps."""
    def patch(tr):
        orig, fired = tr._apply, set()

        def apply(grads):
            norm = orig(grads)
            if tr.step in steps and tr.step not in fired:
                fired.add(tr.step)
                return mx.array(float("nan"))
            return norm

        tr._apply = apply
    return patch


def test_rewind_sets_aside_the_abandoned_branch(tmp_path, tok, shards):
    """--resume <older checkpoint>: the newer ones move aside, so a spike rolls back within the new branch."""
    ck = tmp_path / "ckpt" / "x"
    _, res = run(argv(shards, tmp_path, "x", "--save-every", "5", "--milestone-frac", "0"), tok, stop_at=30)
    assert res == STOPPED and [d.name for d in C._full_dirs(ck)] == ["step_000020", "step_000025", "step_000030"]
    tr, res = run(argv(shards, tmp_path, "x", "--save-every", "2", "--milestone-frac", "0", "--resume",
                       str(ck / "step_000020")), tok, stop_at=27, patch=nan_at(24))
    assert res == STOPPED
    assert [m["step"] for m in metrics(tmp_path, "x", "recover")] == [24]  # back to 24 (this branch), not 30
    assert [d.name for d in C._full_dirs(ck)] == ["step_000024", "step_000026", "step_000027"]
    aside = list(ck.glob("abandoned_*/step_*"))
    assert sorted(p.name for p in aside) == ["step_000025", "step_000030"]
    assert C.latest(ck).name == "step_000027"


def test_no_rollback_target_inside_a_spike_streak(tmp_path, tok, shards):
    """Saves scheduled during a spike streak wait, so the rollback skips the streak's updates."""
    def spikes(tr):
        orig = tr._apply

        def apply(grads):
            norm = orig(grads)
            return mx.array(50.0) if tr.step in (12, 13, 14) and not tr.stats["recoveries"] else norm

        tr._apply = apply

    tr, res = run(argv(shards, tmp_path, "y", "--max-steps", "24", "--spike-norm", "10", "--switch-grace", "0",
                       "--save-every", "5", "--first-save", "13", "--milestone-frac", "0"), tok, patch=spikes)
    assert res == DONE and [m["step"] for m in metrics(tmp_path, "y", "recover")] == [10]
    before, after = (tmp_path / "logs" / "y" / "train.log").read_text().split("RECOVER")
    assert "saved step_000010" in before and "saved step_000013" not in before and "saved step_000014" not in before
    assert "saved step_000013" in after  # the deferred first save, now on the recovered branch


def test_skip_steps_again_after_a_second_stop(tmp_path, tok, shards):
    """The STOP line's command skips data each time (its --lr-scale is new); re-running it after a crash does not."""
    base = argv(shards, tmp_path, "z", "--save-every", "5", "--auto-recover", "0", "--milestone-frac", "0")
    with pytest.raises(SystemExit) as e:
        run(base, tok, patch=nan_at(12))
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        run(base + ["--resume", "auto", "--lr-scale", "0.7", "--skip-steps", "3"], tok, patch=nan_at(17))
    assert e.value.code == 2
    tr, res = run(base + ["--resume", "auto", "--lr-scale", "0.49", "--skip-steps", "3"], tok, stop_at=21)
    assert res == STOPPED and tr.stats["skipped_steps"] == 6 and tr.lr_scale == pytest.approx(0.49)
    tr, res = run(base + ["--resume", "auto", "--lr-scale", "0.49", "--skip-steps", "3"], tok)  # same command again
    assert res == DONE and tr.stats["skipped_steps"] == 6


def test_phase_data_needs_are_checked(tmp_path, tok, shards):
    """A phase that needs more tokens than its shards hold is refused (the loader would repeat it)."""
    with pytest.raises(SystemExit, match="S0 needs .*--allow-repeat"):
        Trainer(build_parser().parse_args(argv(shards, tmp_path, "d", "--max-steps", "400")), tok=tok)
    tr = Trainer(build_parser().parse_args(argv(shards, tmp_path, "d", "--max-steps", "400", "--allow-repeat")), tok=tok)
    assert "repeat data" in (tmp_path / "logs" / "d" / "train.log").read_text() and tr.plan.total == 400


def test_s0_gate_uses_closing_accuracy(tmp_path, tok, shards, monkeypatch):
    """Plan 14 gate on k-Dyck and shuffle-Dyck closing accuracy; the valid rates only with their baselines."""
    import bonsai.train as T
    res = {"kdyck_close_acc": 0.99, "shuffle_close_acc": 0.80, "kdyck_valid_rate": 0.99, "shuffle_valid_rate": 0.97,
           "shuffle_valid_baseline": 0.9406}
    monkeypatch.setattr(T, "optional_import", lambda module, name: (lambda model, tok: res))
    tr = Trainer(build_parser().parse_args(argv(shards, tmp_path, "g")), tok=tok)
    for close, verdict in ((0.80, "shuffle-Dyck closing 80.0% -> BELOW TARGET"), (0.96, "-> PASS")):
        res["shuffle_close_acc"] = close
        tr.run_dyck()
        line = [l for l in (tmp_path / "logs" / "g" / "train.log").read_text().splitlines() if "S0 gate" in l][-1]
        assert verdict in line and "shuffle 97.0% (constant predictor 94.1%)" in line and "kdyck 99.0%;" not in line


def test_ckpt_check_at_the_s0_switch(tmp_path, tok, shards, monkeypatch, capsys):
    """The checkpoint written at the S0 -> S1 switch: no S1 loss was logged before it, so ckpt_check recomputes
    the val loss logged at that step instead of comparing the first S1 loss with an S0 loss."""
    import sys
    import bonsai.tokenizer
    from scripts import ckpt_check
    _, res = run(argv(shards, tmp_path, "cc"), tok, stop_at=10)
    assert res == STOPPED
    monkeypatch.setattr(bonsai.tokenizer, "get_tokenizer", lambda: tok)
    monkeypatch.setattr(sys, "argv", ["ckpt_check", "--run", "cc", "--steps", "3", "--checkpoint",
                                      str(tmp_path / "ckpt" / "cc" / "step_000010")])
    ckpt_check.main()
    out = capsys.readouterr().out
    assert "val loss at step 10" in out and "RECOVERABLE" in out


def test_s0_ablation(tmp_path, tok, shards):
    from scripts.s0_ablation import MAIN, ablation, build_parser as ablation_parser, probe_args, probe_schedule
    assert MAIN["steps"] == 10758
    assert probe_schedule(763) == {"warmup": 21, "muon_warmup": 21, "transition_steps": 14}  # the 100M-token probe
    # the default probe: 300 / 300 / 200 of the main run's 10,758 steps -> 21 / 21 / 14 of its 763 (S0 mostly at peak)
    args, _ = probe_args(ablation_parser().parse_known_args([])[0], [], "s0", 0.05)
    assert (args.warmup, args.muon_warmup, args.transition_steps, args.run) == (21, 21, 14, "bonsai-0.1-probe-s0")
    out = tmp_path / "s0_ablation.json"
    opts, rest = ablation_parser().parse_known_args(
        ["--model", "tiny", "--s0-frac", "0.25", "--final-eval-tokens", "4096", "--out", str(out)]
        + argv(shards, tmp_path, "ignored")[2:])  # drop "--model tiny"; "--run ignored" must not leak through
    res = ablation(opts, rest, tok=tok)
    assert json.loads(out.read_text())["keep_s0"] == res["keep_s0"]
    r0, r1 = res["runs"]["s0"], res["runs"]["nos0"]
    assert (r0["run"], r1["run"]) == ("bonsai-0.1-probe-s0", "bonsai-0.1-probe-nos0")
    assert r0["plan"]["s0_end"] == 10 and r1["plan"]["s0_end"] == 0 and r0["steps"] == r1["steps"] == 40
    assert r0["tokens"] == r1["tokens"] and r0["val_targets"] == r1["val_targets"] > 0
    assert res["delta_nat"] == pytest.approx(r1["val_loss"] - r0["val_loss"])
    assert not (tmp_path / "ckpt" / "ignored").exists()
    # --warmup / --transition-steps given: used as is; the Muon momentum warmup is scaled (40 steps -> 1)
    assert res["schedule"] == {"steps": 40, "warmup": 5, "muon_warmup": 1, "transition_steps": 6,
                               "scaled_from_main_run": MAIN}
    assert r0["plan"]["warmup"] == 5 and r0["plan"]["transition"] == 24
