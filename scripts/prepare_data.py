"""Step 2-4 for Bonsai's new pools: filter, dedup, decontaminate (plan sections 10, 17; DESIGN 1.1).

    python -m scripts.prepare_data                       # nemotron_cc + smoltalk2_think
    python -m scripts.prepare_data --only smoltalk2_think

pagoda's pools are symlinked by `scripts.download --only link`; the formal and identity pools come from
scripts.gen_formal / scripts.gen_identity. Counts go to raw/filtered/stats_data.json.
"""

import argparse

from bonsai.data.prepare import PREPS, run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(PREPS))
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    run(only=args.only, workers=args.workers or None)


if __name__ == "__main__":
    main()
