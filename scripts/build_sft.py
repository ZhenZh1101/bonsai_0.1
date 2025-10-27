"""Off-policy SFT set + identity mini-SFT set (plan sections 11, 18) -> tok/offpolicy, tok/identity_sft.

    python -m scripts.build_sft                           # configs/sft.yaml
    python -m scripts.build_sft --config configs/smoke/sft.yaml
    python -m scripts.build_sft --no-identity             # without identity conversations (and no identity_sft)

Reads the kept teacher samples (data/offpolicy/kept_*.jsonl) and raw/filtered/identity_chat.jsonl; see
bonsai/sft.py for the format, the identity share and the mode signalling. The report goes into the train
manifest and logs/build_sft.json.
"""

import argparse

from bonsai.common import CONFIG_DIR, LOG_DIR, load_yaml, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "sft.yaml"))
    ap.add_argument("--no-identity", action="store_true")
    args = ap.parse_args()
    from bonsai.sft import build_offpolicy_sft
    from bonsai.tokenizer import get_tokenizer
    cfg = load_yaml(args.config)
    report = build_offpolicy_sft(cfg, get_tokenizer(), identity_ok=not args.no_identity,
                                 log_file=LOG_DIR / "build_sft.log")
    write_json(LOG_DIR / f"build_sft_{cfg.get('shard_dir', 'offpolicy').replace('/', '_')}.json", report)


if __name__ == "__main__":
    main()
