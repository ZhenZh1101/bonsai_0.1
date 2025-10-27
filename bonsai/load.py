"""Load a trained Bonsai model for evaluation, chat, export or the next training stage (plan section 18 names).

    load_model(run, source) -> (model, tok, meta)

source:
    base       ckpt/<run>/base.safetensors (Bonsai-0.1-base, end of S2), else the latest full checkpoint
    latest     the latest full checkpoint of ckpt/<run>/ (during pretraining)
    sft        ckpt/<run>/sft_final.safetensors (Bonsai-0.1-sft, after off-policy distillation)
    onpolicy   ckpt/<run>/onpolicy_final.safetensors (after on-policy distillation)
    final      ckpt/<run>/final.safetensors (Bonsai-0.1, the release: after the identity mini-SFT)
    <path>     a full checkpoint dir or a .safetensors weights file

Weights files carry meta["model_config"] and meta["vocab_sha"]; a vocab_sha that differs from the tokenizer's
is refused (the embedding rows would mean other tokens). meta["path"] is the resolved file / dir.
"""

from pathlib import Path

import mlx.core as mx

from bonsai import checkpoint as C
from bonsai.common import CKPT_DIR
from bonsai.model import build_model

FILES = {"sft": "sft_final.safetensors", "onpolicy": "onpolicy_final.safetensors", "final": "final.safetensors"}
HINT = {"sft": "scripts.sft", "onpolicy": "scripts.distill_onpolicy", "final": "scripts.identity_sft"}


def resolve(run, source="base", ckpt_dir=CKPT_DIR):
    run_dir = Path(ckpt_dir) / run if run else None
    if source in ("base", "latest"):
        if source == "base" and run_dir is not None and (run_dir / "base.safetensors").exists():
            return run_dir / "base.safetensors"
        p = C.latest(run_dir) if run_dir is not None else None
        if p is None:
            raise FileNotFoundError(f"no base.safetensors or full checkpoint in {run_dir}")
        return p
    if source in FILES:
        if run_dir is None:
            raise ValueError(f"source {source!r} needs a run")
        p = run_dir / FILES[source]
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (run {HINT[source]} first)")
        return p.resolve()
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (source: base | latest | sft | onpolicy | final | a path)")
    return p


def load_model(run="bonsai-0.1", source="base", tok=None, dtype=mx.bfloat16, ckpt_dir=CKPT_DIR):
    """(model with fp32 weights computing in `dtype`, student tokenizer, meta)."""
    path = resolve(run, source, ckpt_dir)
    meta = C.load_meta(path)
    if "model_config" not in meta:
        raise ValueError(f"{path} has no meta.model_config")
    if tok is None:
        from bonsai.tokenizer import get_tokenizer
        tok = get_tokenizer()
    sha = meta.get("vocab_sha")
    if sha != tok.vmap.sha():
        raise ValueError(f"{path} was trained with vocabulary subset {sha!r}, the tokenizer has {tok.vmap.sha()!r} "
                         f"(tokenizer/vocab_subset.npz changed?)")
    model = build_model(meta["model_config"], compute_dtype=dtype)
    C.load_weights(model, path)
    meta["path"] = str(path)
    return model, tok, meta
