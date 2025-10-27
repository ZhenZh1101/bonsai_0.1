"""Checkpoints with rotation (plan section 8: every 30 min, newest 3 full + a bf16 weight milestone every 10%).

ckpt/<run>/step_000123/           full checkpoint: fp32 weights, optimizer state, meta.json (step, plan, loader
    model.safetensors             state of every phase, model config, vocab_sha, optimizer config, loop stats)
    optim.safetensors
    meta.json
ckpt/<run>/milestones/step_000123.bf16.safetensors    weights only (metadata "meta" = the same meta as JSON)
ckpt/<run>/base.safetensors       end of pretraining = Bonsai-0.1-base (bf16, metadata "meta")

Only the newest `keep` full checkpoints are kept. Writes go to a temp dir that is renamed into place, so a crash
mid-save never leaves a half-written checkpoint. Weights files always carry meta["model_config"] and
meta["vocab_sha"] (bonsai.load checks the latter against the tokenizer). A run resumed from an older checkpoint
moves the newer ones (the abandoned branch) to ckpt/<run>/abandoned_<time>/ (`set_aside_newer`), so rotation and
`latest` only ever see the branch being trained.
"""

import json
import os
import shutil
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def _full_dirs(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("step_") and (d / "meta.json").exists()]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[1]))


def latest(run_dir):
    dirs = _full_dirs(run_dir)
    return dirs[-1] if dirs else None


def set_aside_newer(run_dir, step):
    """Move the full checkpoints of `run_dir` newer than `step` into run_dir/abandoned_<time>/ (nothing is
    deleted) and return their new paths. Called when training (re)starts at `step`: otherwise rotation would
    delete the new branch's checkpoints and `latest` (--resume auto, spike recovery) would jump back to the old one."""
    newer = [d for d in _full_dirs(run_dir) if int(d.name.split("_")[1]) > step]
    if not newer:
        return []
    dest = Path(run_dir) / f"abandoned_{time.strftime('%Y%m%d-%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    for d in newer:
        os.replace(d, dest / d.name)
    return [dest / d.name for d in newer]


def save_full(run_dir, step, model, opt, meta, keep=3):
    run_dir = Path(run_dir)
    final = run_dir / f"step_{step:06d}"
    tmp = run_dir / f".tmp_step_{step:06d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "model.safetensors"), dict(tree_flatten(model.parameters())))
    if opt is not None:
        mx.save_safetensors(str(tmp / "optim.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "meta.json").write_text(json.dumps({**meta, "step": step}, indent=2))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    for old in _full_dirs(run_dir)[:-keep]:
        shutil.rmtree(old)
    return final


def save_weights(path, model, meta, dtype=mx.bfloat16):
    """Standalone weights file (milestones, base / sft / onpolicy / final models), written atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {k: v.astype(dtype) for k, v in tree_flatten(model.parameters())}
    tmp = path.with_name(f".tmp_{path.name}")
    mx.save_safetensors(str(tmp), weights, metadata={"meta": json.dumps(meta)})
    os.replace(tmp, path)
    return path


def save_milestone(run_dir, step, model, meta):
    return save_weights(Path(run_dir) / "milestones" / f"step_{step:06d}.bf16.safetensors", model,
                        {**meta, "step": step})


def load_meta(path):
    path = Path(path)
    if path.is_dir():
        return json.loads((path / "meta.json").read_text())
    _, metadata = mx.load(str(path), return_metadata=True)
    return json.loads(metadata.get("meta", "{}"))


def weights_file(path):
    path = Path(path)
    return path / "model.safetensors" if path.is_dir() else path


def load_weights(model, path):
    """Load weights (full checkpoint dir or .safetensors file) into `model` in place as fp32 (strict: names and
    shapes must match the model exactly)."""
    weights = mx.load(str(weights_file(path)))
    model.load_weights([(k, v.astype(mx.float32)) for k, v in weights.items()], strict=True)
    mx.eval(model.parameters())


def load_optimizer(opt, path):
    """Restore optimizer state in place (keeps the dict objects a compiled step captured)."""
    flat = mx.load(str(Path(path) / "optim.safetensors"))
    tree = tree_unflatten(list(flat.items()))
    if len(tree["states"]) != len(opt.optimizers):
        raise ValueError(f"{path}: optimizer has {len(tree['states'])} groups, expected {len(opt.optimizers)} "
                         f"(different --optimizer?)")
    for o, s in zip(opt.optimizers, tree["states"]):
        o.state.clear()
        o.state.update(s)
    mx.eval(opt.state)
