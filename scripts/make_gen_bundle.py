"""Pack everything the off-policy teacher generation needs into one tarball for a second Mac (plan sections 3, 11).

    python -m scripts.make_gen_bundle                              # -> dist/bonsai-gen-<date>.tar.gz (~30 MB)
    python -m scripts.make_gen_bundle --with-teacher Qwen3-0.6B    # also the weights, for a machine without HF access

On the other Mac: `tar xzf bonsai-gen-<date>.tar.gz && cd bonsai-gen && ./run_gen.sh` (see its README.md).

Contents: the bonsai/ package, scripts/, configs/, identity/ (Qwen / other-model self-claim filter), the Qwen3
tokenizer files + vocab subset + hashes, data/prompts.jsonl (the 100k off-policy prompts), the eval test sets the
13-gram decontamination of teacher outputs needs (GSM8K, ARC, HellaSwag, MMLU, HumanEval, FLD), pyproject.toml +
uv.lock (the same package versions as here) and the runs/remote_gen/ scripts at the top. BUNDLE.json records the
sha256 of the prompts and the config, so scripts.merge_offpolicy can check that results match this machine.
"""

import argparse
import hashlib
import json
import shutil
import socket
import tarfile
import time
from pathlib import Path

from bonsai.common import CODE_ROOT, DATA_ROOT, RAW_DIR, TEACHER_DIR, TOKENIZER_DIR, log
from bonsai.tokenizer import TOKENIZER_FILES

EVAL_FILES = ["gsm8k/main/test-00000-of-00001.parquet", "arc/ARC-Easy/test-00000-of-00001.parquet",
              "arc/ARC-Challenge/test-00000-of-00001.parquet", "hellaswag/data/validation-00000-of-00001.parquet",
              "mmlu/all/test-00000-of-00001.parquet", "humaneval/openai_humaneval/test-00000-of-00001.parquet"]
SKIP = {"__pycache__", ".DS_Store", ".pytest_cache"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src, dst, suffixes=None):
    n = 0
    for p in sorted(Path(src).rglob("*")):
        rel = p.relative_to(src)
        if any(part in SKIP for part in rel.parts) or not p.is_file():
            continue
        if suffixes and p.suffix not in suffixes:
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p.resolve(), dst / rel)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DATA_ROOT / "dist"))
    ap.add_argument("--with-teacher", nargs="*", default=[], help="include teachers/<name> (Qwen3-0.6B, Qwen3-1.7B)")
    args = ap.parse_args()
    out = Path(args.out)
    root = out / "bonsai-gen"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    files = copy_tree(CODE_ROOT / "bonsai", root / "bonsai", {".py"})
    files += copy_tree(CODE_ROOT / "scripts", root / "scripts", {".py"})
    files += copy_tree(CODE_ROOT / "configs", root / "configs", {".yaml"})
    files += copy_tree(CODE_ROOT / "identity", root / "identity")
    for f in ("pyproject.toml", "uv.lock"):
        shutil.copy2(CODE_ROOT / f, root / f)
    for f in sorted(p.name for p in (CODE_ROOT / "runs" / "remote_gen").iterdir() if p.is_file()):
        shutil.copy2(CODE_ROOT / "runs" / "remote_gen" / f, root / f)
    (root / "tokenizer").mkdir()
    for f in list(TOKENIZER_FILES) + ["manifest.json", "vocab_subset.npz"]:
        shutil.copy2(TOKENIZER_DIR / f, root / "tokenizer" / f)
    (root / "data").mkdir()
    prompts = DATA_ROOT / "data" / "prompts.jsonl"
    shutil.copy2(prompts, root / "data" / "prompts.jsonl")
    evals = EVAL_FILES + [str(p.relative_to(RAW_DIR)) for p in sorted((RAW_DIR / "fld_v2").glob("*/test-*.parquet"))]
    for rel in evals:
        (root / "raw" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((RAW_DIR / rel).resolve(), root / "raw" / rel)
    for name in args.with_teacher:
        log(f"including teachers/{name}")
        copy_tree(TEACHER_DIR / name, root / "teachers" / name)

    cfg = CODE_ROOT / "configs" / "offpolicy.yaml"
    n_prompts = sum(1 for _ in open(prompts))
    bundle = {"created": time.strftime("%F %T"), "from_host": socket.gethostname(), "prompts_sha256": sha256(prompts),
              "prompts": n_prompts, "offpolicy_config_sha256": sha256(cfg),
              "tokenizer": json.loads((TOKENIZER_DIR / "manifest.json").read_text())["sha256"],
              "teachers_included": args.with_teacher, "code_files": files}
    (root / "BUNDLE.json").write_text(json.dumps(bundle, indent=2))
    tar_path = out / f"bonsai-gen-{time.strftime('%Y%m%d-%H%M')}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(root, arcname="bonsai-gen")
    log(f"bundle: {files} code files, {n_prompts:,} prompts, {len(evals)} eval files, teachers {args.with_teacher or 'none'} "
        f"-> {tar_path} ({tar_path.stat().st_size / 1e6:.0f} MB)")
    log(f"on the other Mac: tar xzf {tar_path.name} && cd bonsai-gen && ./run_gen.sh")


if __name__ == "__main__":
    main()
