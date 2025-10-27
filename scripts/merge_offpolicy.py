"""Bring teacher-generation results from another Mac into data/offpolicy (plan section 11; runs/remote_gen/).

    python -m scripts.merge_offpolicy results/bonsai-offpolicy-<host>-<teacher>-s0of1-<date>.tar.gz [more.tar.gz ...]
    python -m scripts.merge_offpolicy <tarball> --dry-run       # only check and report

Checks before anything is copied: the tarball holds only regular files under the expected names; its prompts
are this machine's data/prompts.jsonl (sha256 in BUNDLE.json, --force to skip) and every sample id is a prompt
of it; a shard file that already exists here is replaced only by one with at least as many samples (a later
pack of the same run); no sample id is in two different shard files (e.g. s0of1 next to s0of2). Then
data/offpolicy/stats.json is recomputed from all kept / rejected files (bonsai.teacher.summarize).
runs/after_pretrain.sh rebuilds tok/offpolicy (scripts.build_sft) because the kept files are newer.
"""

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from bonsai.common import DATA_ROOT, iter_jsonl, log

ALLOWED = re.compile(r"^bonsai-gen/(BUNDLE\.json|data/remote_plan\.json|logs/gen_offpolicy\.log|"
                     r"data/offpolicy/(kept|rejected)_s\d+of\d+\.jsonl|data/offpolicy/(runs\.jsonl|stats\.json))$|"
                     r"^(BUNDLE\.json|data/remote_plan\.json|logs/gen_offpolicy\.log|"
                     r"data/offpolicy/(kept|rejected)_s\d+of\d+\.jsonl|data/offpolicy/(runs\.jsonl|stats\.json))$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def samples(path):
    """Ids of a kept / rejected file; a cut last line (hard kill before packing) is ignored, as gen_offpolicy does."""
    ids = []
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            ids.append(json.loads(line)["id"])
    return ids


def extract(tar_path, tmp):
    with tarfile.open(tar_path) as tar:
        # macOS tar may add AppleDouble metadata (._name): not data, skip it
        members = [m for m in tar.getmembers() if not m.isdir() and not Path(m.name).name.startswith("._")]
        bad = [m.name for m in members if not m.isfile() or not ALLOWED.match(m.name)]
        if bad:
            raise SystemExit(f"{tar_path}: unexpected entries {bad[:5]} (refusing to extract)")
        for m in members:
            dest = tmp / m.name.removeprefix("bonsai-gen/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return tmp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tarballs", nargs="+")
    ap.add_argument("--dest", default=str(DATA_ROOT / "data" / "offpolicy"))
    ap.add_argument("--prompts", default=str(DATA_ROOT / "data" / "prompts.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="accept results made from a different prompts.jsonl")
    args = ap.parse_args()
    dest = Path(args.dest)
    prompt_ids = {r["id"] for r in iter_jsonl(args.prompts)}
    local_sha = sha256(args.prompts)

    # what is already here, by file
    have = {p.name: samples(p) for p in sorted(dest.glob("*_s*of*.jsonl"))} if dest.exists() else {}
    plan = dict(have)
    with tempfile.TemporaryDirectory() as td:
        staged = []
        for k, tb in enumerate(args.tarballs):
            src = extract(tb, Path(td) / str(k))
            bundle = json.loads((src / "BUNDLE.json").read_text()) if (src / "BUNDLE.json").exists() else {}
            if bundle.get("prompts_sha256") != local_sha and not args.force:
                raise SystemExit(f"{tb}: made from another data/prompts.jsonl (bundle {bundle.get('prompts_sha256', '?')[:12]} "
                                 f"vs here {local_sha[:12]}); rebuild the bundle or pass --force")
            rp = json.loads((src / "data/remote_plan.json").read_text()) if (src / "data/remote_plan.json").exists() else {}
            for p in sorted((src / "data/offpolicy").glob("*_s*of*.jsonl")):
                ids = samples(p)
                unknown = [i for i in ids if i.rsplit(":", 1)[0] not in prompt_ids]
                if unknown:
                    raise SystemExit(f"{tb}: {p.name} has {len(unknown)} ids that are not prompts here, e.g. {unknown[:3]}")
                old = plan.get(p.name)
                if old is not None and len(old) > len(ids):
                    log(f"  {p.name}: keeping the local file ({len(old):,} samples > {len(ids):,} in {Path(tb).name})")
                    continue
                plan[p.name] = ids
                staged.append((p, p.name, len(ids), old is not None))
            runs = src / "data/offpolicy/runs.jsonl"
            if runs.exists():
                staged.append((runs, "runs.jsonl", None, None))
            log(f"{Path(tb).name}: {rp.get('host', '?')}, teacher {rp.get('teacher', '?')}, shard {rp.get('shard', '?')}")
        seen = {}
        for name, ids in plan.items():
            for i in ids:
                if i in seen and seen[i] != name:
                    raise SystemExit(f"sample {i} is in both {seen[i]} and {name}: different shard splits overlap")
                seen[i] = name
        kept = sum(len(v) for k, v in plan.items() if k.startswith("kept_"))
        rej = sum(len(v) for k, v in plan.items() if k.startswith("rejected_"))
        for p, name, n, replaced in staged:
            if n is not None:
                log(f"  {name}: {n:,} samples{' (replaces the local file)' if replaced else ''}")
        log(f"after merge: {kept:,} kept + {rej:,} rejected = {kept + rej:,} of {2 * len(prompt_ids):,} jobs "
            f"({(kept + rej) / max(2 * len(prompt_ids), 1):.1%})")
        if args.dry_run:
            log("dry run: nothing copied")
            return
        dest.mkdir(parents=True, exist_ok=True)
        run_lines = set()
        if (dest / "runs.jsonl").exists():
            run_lines = {l for l in open(dest / "runs.jsonl") if l.strip()}
        for p, name, n, _ in staged:
            if name == "runs.jsonl":
                new = [l for l in open(p) if l.strip() and l not in run_lines]
                with open(dest / "runs.jsonl", "a") as f:
                    f.writelines(new)
                run_lines.update(new)
            else:
                tmp = dest / f".{name}.tmp"
                shutil.copy2(p, tmp)
                tmp.replace(dest / name)
    from bonsai.teacher import summarize
    st = summarize(dest)
    log(f"merged into {dest}: kept {st['kept']:,} / {st['jobs']:,} ({st['kept_frac']:.0%}); by kind/mode {st['kept_by']}")
    log("next: runs/after_pretrain.sh rebuilds tok/offpolicy from these files (or run python -m scripts.build_sft)")


if __name__ == "__main__":
    main()
