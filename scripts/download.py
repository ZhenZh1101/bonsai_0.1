"""Step 1 of the data pipeline: fetch what Bonsai needs (plan sections 3, 9, 10, 17).

    python -m scripts.download                       # everything below
    python -m scripts.download --only tokenizer teachers
    python -m scripts.download --only link           # symlink pagoda's raw data and filtered pools
    python -m scripts.download --only nemotron procedural_pile fld_v2

  tokenizer   Qwen3 tokenizer files at the pinned commit -> tokenizer/ (+ sha256 manifest, plan section 6)
  teachers    Qwen3-0.6B and Qwen3-1.7B at pinned commits -> teachers/ (mlx-lm loads them from there);
              --teachers Qwen3-0.6B fetches one of them
  link        link Pagoda downloads into raw/<name> and filtered pools into raw/filtered/<pool>.jsonl
              (only needed to rebuild the shards; the eval sets are Bonsai's own, see evals)
  evals       GSM8K / MMLU / ARC / HellaSwag / HumanEval test sets -> raw/<name>/ (evals, decontamination)
  core        the CORE eval bundle -> raw/eval_bundle/
  nemotron    Nemotron-CC synthetic subsets streamed from Common Crawl -> raw/nemotron_cc/<kind>.jsonl
  <hf name>   files listed in bonsai/data/sources.py HF_FILES -> raw/<name>/

Re-running is cheap: files already present are skipped.
"""

import argparse
import gzip
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from bonsai.common import FILTERED_DIR, PAGODA_ROOT, RAW_DIR, TEACHER_DIR, TOKENIZER_DIR, ensure_dirs, log, write_json
from bonsai.data import sources as S
from bonsai.tokenizer import QWEN_REPO, QWEN_REVISION, TOKENIZER_FILES, sha256

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _retry(fn, what, attempts=5):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # network hiccups: retry with backoff
            if attempt == attempts - 1:
                raise
            log(f"retry {what}: {e}")
            time.sleep(5 * (attempt + 1))


def download_tokenizer():
    from huggingface_hub import hf_hub_download
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    for f in TOKENIZER_FILES:
        _retry(lambda: hf_hub_download(QWEN_REPO, f, revision=QWEN_REVISION, local_dir=TOKENIZER_DIR), f)
    write_tokenizer_manifest()


def write_tokenizer_manifest():
    files = {f: sha256(TOKENIZER_DIR / f) for f in TOKENIZER_FILES}
    teachers = {}
    for name in S.TEACHERS:
        d = TEACHER_DIR / name
        if all((d / f).exists() for f in TOKENIZER_FILES):
            teachers[name] = {f: sha256(d / f) for f in TOKENIZER_FILES}
    mismatch = {t: [f for f in TOKENIZER_FILES if h[f] != files[f]] for t, h in teachers.items()}
    mismatch = {t: m for t, m in mismatch.items() if m}
    write_json(TOKENIZER_DIR / "manifest.json", {"repo": QWEN_REPO, "revision": QWEN_REVISION, "sha256": files,
                                                 "teachers": teachers, "mismatch": mismatch})
    if mismatch:
        raise SystemExit(f"tokenizer files differ between the student copy and teachers: {mismatch}")
    log(f"tokenizer: {QWEN_REPO}@{QWEN_REVISION[:8]}, hashes identical for student + {sorted(teachers)}")


def download_teachers(names=None):
    from huggingface_hub import snapshot_download
    for name, (repo, rev) in S.TEACHERS.items():
        if names and name not in names:
            continue
        dest = TEACHER_DIR / name
        _retry(lambda: snapshot_download(repo, revision=rev, local_dir=dest, max_workers=4), repo)
        log(f"teacher {name}: {repo}@{rev[:8]} -> {dest}")
    if (TOKENIZER_DIR / "tokenizer.json").exists():
        write_tokenizer_manifest()


def link_pagoda():
    """Symlink pagoda's raw downloads and filtered pools (plan: reuse pagoda's open-source data)."""
    if not PAGODA_ROOT.exists():
        raise SystemExit(f"{PAGODA_ROOT} not found (set PAGODA_HOME)")
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in S.PAGODA_RAW:
        src, dst = PAGODA_ROOT / "raw" / name, RAW_DIR / name
        if not src.exists():
            log(f"warning: pagoda raw/{name} missing")
            continue
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src)
            n += 1
    for pool in S.PAGODA_POOLS:
        src, dst = PAGODA_ROOT / "raw" / "filtered" / f"{pool}.jsonl", FILTERED_DIR / f"{pool}.jsonl"
        if not src.exists():
            log(f"warning: pagoda pool {pool} missing")
            continue
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src)
            n += 1
    stats = PAGODA_ROOT / "raw" / "filtered" / "stats.json"
    if stats.exists():
        write_json(FILTERED_DIR / "pagoda_stats.json", json.loads(stats.read_text()))
    log(f"linked {n} new pagoda paths into {RAW_DIR}")


def drop_dead_link(path):
    """Replace a dangling raw/<name> symlink with a real directory on the next download."""
    if path.is_symlink() and not path.exists():
        path.unlink()
        log(f"removed dead link {path}")


def download_hf(names, workers, table=None):
    from huggingface_hub import hf_hub_download
    table = table or S.HF_FILES
    jobs = [(name, repo, rev, f) for name, (repo, rev, files) in table.items() if name in names for f in files]
    for name in {j[0] for j in jobs}:
        drop_dead_link(RAW_DIR / name)

    def fetch(job):
        name, repo, rev, filename = job
        dest = RAW_DIR / name / filename
        if dest.exists() and dest.stat().st_size > 0:
            return name, filename, dest.stat().st_size, "cached"
        path = _retry(lambda: hf_hub_download(repo, filename, repo_type="dataset", revision=rev,
                                              local_dir=RAW_DIR / name), f"{repo}/{filename}")
        return name, filename, os.path.getsize(path), "ok"

    total = 0
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(fetch, j) for j in jobs]):
            name, filename, size, status = fut.result()
            total += size
            log(f"{status:6s} {name}/{filename}  {size / 1e6:.0f} MB")
    log(f"HF files: {len(jobs)} files, {total / 1e9:.2f} GB")


def download_core_bundle():
    import zipfile
    dest = RAW_DIR / "eval_bundle"
    drop_dead_link(dest)
    if (dest / "core.yaml").exists():
        log(f"cached {dest}")
        return
    zpath = RAW_DIR / "eval_bundle.zip"
    log(f"downloading {S.CORE_BUNDLE_URL}")
    _retry(lambda: urllib.request.urlretrieve(S.CORE_BUNDLE_URL, zpath), S.CORE_BUNDLE_URL)
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():  # refuse path traversal
            if member.startswith("/") or ".." in member.split("/"):
                raise SystemExit(f"suspicious path in zip: {member}")
        z.extractall(RAW_DIR)
    zpath.unlink()
    log(f"CORE bundle extracted to {dest}")


def _nemotron_parts():
    with urllib.request.urlopen(S.NEMOTRON_PATHS, timeout=60) as r:
        paths = gzip.decompress(r.read()).decode().split()
    out = {}
    for kind in S.NEMOTRON_KINDS:
        picks = []
        for crawl in S.NEMOTRON_CRAWLS:
            cands = sorted(p for p in paths if f"quality=high/kind=synthetic/kind2={kind}/" in p and f"/{crawl}-" in p)
            if cands:
                picks.append(cands[len(cands) // 2])
        out[kind] = picks
    return out


def download_nemotron():
    """Stream Nemotron-CC (v1) high-quality synthetic parts and keep the first N characters of each."""
    import zstandard
    out_dir = RAW_DIR / "nemotron_cc"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = _nemotron_parts()
    for kind, plist in parts.items():
        dest = out_dir / f"{kind}.jsonl"
        if dest.exists() and dest.stat().st_size > 0:
            log(f"cached {dest}")
            continue
        tmp = dest.with_suffix(".tmp")
        docs = chars = 0
        with open(tmp, "w") as w:
            for path in plist:
                got = 0

                def stream():
                    nonlocal got
                    got = 0
                    req = urllib.request.Request(S.NEMOTRON_BASE + path)
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        reader = zstandard.ZstdDecompressor().stream_reader(resp)
                        text = io.TextIOWrapper(reader, encoding="utf-8")
                        rows = []
                        for line in text:
                            rec = json.loads(line)
                            rows.append({"text": rec["text"], "url": rec.get("url"), "kind": kind,
                                         "part": path.rsplit("/", 1)[-1]})
                            got += len(rec["text"])
                            if got >= S.NEMOTRON_CHARS_PER_PART:
                                break
                        return rows

                rows = _retry(stream, path)
                for r in rows:
                    w.write(json.dumps(r, ensure_ascii=False) + "\n")
                docs += len(rows)
                chars += got
                log(f"  nemotron {kind}: {path.rsplit('/', 1)[-1]} -> {len(rows):,} docs, {got / 1e6:.0f}M chars")
        os.replace(tmp, dest)
        log(f"nemotron {kind}: {docs:,} docs, {chars / 1e6:.0f}M chars -> {dest}")
    write_json(out_dir / "parts.json", parts)


def write_manifest():
    files = {}
    for p in sorted(RAW_DIR.iterdir()):
        if p.is_symlink() or p.name == "filtered":
            continue
        for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
            if f.is_file() and ".cache" not in f.parts:
                files[str(f.relative_to(RAW_DIR))] = f.stat().st_size
    links = {p.name: str(os.readlink(p)) for p in sorted(RAW_DIR.iterdir()) if p.is_symlink()}
    write_json(RAW_DIR / "MANIFEST.json", {"total_bytes": sum(files.values()), "files": files, "pagoda_links": links})
    log(f"raw/ (Bonsai's own downloads) {sum(files.values()) / 1e9:.2f} GB in {len(files)} files, "
        f"{len(links)} pagoda links")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help=f"tokenizer teachers link evals core nemotron {' '.join(S.HF_FILES)}")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--teachers", nargs="*", help=f"with --only teachers: just these of {list(S.TEACHERS)}")
    args = ap.parse_args()
    ensure_dirs()
    only = set(args.only) if args.only else None

    def want(x):
        return only is None or x in only

    if want("tokenizer"):
        download_tokenizer()
    if want("link"):
        link_pagoda()
    hf = [n for n in S.HF_FILES if want(n)]
    if hf:
        download_hf(hf, args.workers)
    if want("evals"):
        download_hf(list(S.EVAL_FILES), args.workers, S.EVAL_FILES)
    if want("core"):
        download_core_bundle()
    if want("nemotron"):
        download_nemotron()
    if want("teachers"):
        download_teachers(args.teachers)
    write_manifest()


if __name__ == "__main__":
    main()
