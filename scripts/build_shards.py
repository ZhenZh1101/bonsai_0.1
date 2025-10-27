"""Steps 5-6: mix, tokenize and shard the pretraining stages (plan sections 9, 10; DESIGN 1.2, 1.3).

    python -m scripts.build_shards                     # s1 (+ tok/val) -> s2 -> s0 (+ tok/val_s0)
    python -m scripts.build_shards --only s2           # rebuild S2 alone, starting where S1 left the pools
    python -m scripts.build_shards --smoke             # configs/smoke/*.yaml -> tok/smoke/...
    python -m scripts.build_shards --smoke --skip-missing   # leave out sources whose pool is not there yet

S1 and S2 share pools through tok/pool_cursors.json (smoke: tok/smoke/pool_cursors.json); building S1 resets
the cursors, so after rebuilding S1 always rebuild S2 (and the prompt pool) too. Rebuilding S2 alone first
resets the cursors to the end of S1 (tok/s1/manifest.json). S0 reads its own pools from line 0. Shards are in
student ids of the current tokenizer/vocab_subset.npz (manifest vocab_sha): rebuild everything after changing
the subset. Then run `python -m scripts.verify_data`.
"""

import argparse

from bonsai.common import CONFIG_DIR, TOK_DIR, load_yaml, log
from bonsai.data.build import RenderPool, build_stage, restore_cursors_after
from bonsai.tokenizer import Tokenizer

STAGES = ("s1", "s2", "s0")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=STAGES)
    ap.add_argument("--smoke", action="store_true", help="use configs/smoke/*.yaml")
    ap.add_argument("--config-dir", default=None, help="directory with s0.yaml / s1.yaml / s2.yaml")
    ap.add_argument("--total-tokens", type=float, default=None, help="override total_tokens (timing runs)")
    ap.add_argument("--shard-dir", default=None, help="override shard_dir / val_dir prefix (timing runs)")
    ap.add_argument("--skip-missing", action="store_true", help="drop sources whose pool is missing")
    ap.add_argument("--workers", type=int, default=0, help="render processes (default: cores - 2)")
    args = ap.parse_args()
    cdir = args.config_dir or (CONFIG_DIR / "smoke" if args.smoke else CONFIG_DIR)
    tok = Tokenizer.load()
    log(f"tokenizer: N = {tok.vocab_size:,}, vocab_sha {tok.vmap.sha()}, shard dtype {tok.vmap.shard_dtype.__name__}")
    cfgs = {s: load_yaml(f"{cdir}/{s}.yaml") for s in STAGES}
    for cfg in cfgs.values():
        if args.total_tokens:
            cfg["total_tokens"] = args.total_tokens
        if args.shard_dir:
            cfg["shard_dir"] = f"{args.shard_dir}/{cfg['shard_dir'].split('/')[-1]}"
            cfg["val_dir"] = f"{args.shard_dir}/{cfg.get('val_dir', 'val').split('/')[-1]}"
            cfg["cursor_file"] = f"{args.shard_dir}/pool_cursors.json"
    todo = args.only or list(STAGES)
    with RenderPool(tok, args.workers or None) as workers:
        if "s1" in todo:
            log("== S1 (+ val)")
            build_stage(cfgs["s1"], tok, reset_cursors=True, workers=workers, skip_missing=args.skip_missing)
        if "s2" in todo:
            log("== S2")
            if "s1" not in todo:
                if (TOK_DIR / cfgs["s1"]["shard_dir"] / "manifest.json").exists():
                    c = restore_cursors_after(cfgs["s2"], cfgs["s1"]["shard_dir"])
                    log(f"  cursors reset to the end of {cfgs['s1']['shard_dir']}: {c}")
                else:
                    log(f"  warning: {cfgs['s1']['shard_dir']} not built; S2 continues from the cursor file")
            build_stage(cfgs["s2"], tok, workers=workers, skip_missing=args.skip_missing)
        if "s0" in todo:
            log("== S0 (+ val_s0)")
            build_stage(cfgs["s0"], tok, workers=workers, skip_missing=args.skip_missing)


if __name__ == "__main__":
    main()
