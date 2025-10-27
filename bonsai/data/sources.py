"""What Bonsai downloads (plan sections 9, 10 and 17) and what it reuses from pagoda.

Reused from Pagoda (symlinked into raw/, never copied; 14.8 GB of downloads, 18 GB of filtered pools):
FineWeb-Edu, DCLM-baseline, Cosmopedia v2, FineMath-4+, TinyGSM, Stack-Edu Python, SmolTalk / SmolTalk2,
OpenMathInstruct-1/2, GSM8K, MMLU, ARC, HellaSwag, HumanEval and the CORE eval bundle. pagoda's filtered pools
(raw/filtered/*.jsonl) are already 13-gram decontaminated against the same eval sets.

New for Bonsai: the Qwen3 tokenizer + teachers (pinned commits), Nemotron-CC synthetic subsets, the formal-
language / logic sets of S0 and S2, FLD.v2. All files are fetched one by one into raw/<name>/ with
`hf_hub_download(local_dir=...)`, so the Hugging Face cache never holds an Arrow copy.
"""

# pagoda raw/ directories are linked into raw/<name> (only needed to rebuild the shards). The eval /
# decontamination sets are Bonsai's own downloads (EVAL_FILES), so evals and the teacher-output filter do not depend
# on Pagoda; raw/eval_bundle (CORE) is linked when pagoda has it, else downloaded (`--only core`).
PAGODA_RAW = ["cosmopedia_v2", "dclm", "finemath", "fineweb_edu", "openmathinstruct1", "openmathinstruct2", "smoltalk",
              "smoltalk2", "stack_edu", "tinygsm"]

# pagoda filtered pools reused as raw/filtered/<pool>.jsonl (text pools are re-tokenized with Qwen3 at build time;
# pagoda's identity pools are NOT reused: Bonsai builds its own from identity/)
PAGODA_POOLS = ["fineweb_edu", "fineweb_edu_hq", "dclm", "cosmopedia", "cosmopedia_textbook", "finemath", "tinygsm",
                "stack_python", "smoltalk_chat", "omi2_cot", "gsm8k_train", "omi1_code", "mc", "sft_chat"]

# eval sets (plan section 15) and the 13-gram decontamination texts: the files pagoda used, now downloaded into
# raw/<name>/ by Bonsai itself (`scripts.download --only evals`); name -> (repo_id, revision, [files])
EVAL_FILES = {
    "gsm8k": ("openai/gsm8k", "main", ["main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"]),
    "mmlu": ("cais/mmlu", "main", ["all/test-00000-of-00001.parquet", "all/dev-00000-of-00001.parquet",
                                  "all/validation-00000-of-00001.parquet"]),
    "arc": ("allenai/ai2_arc", "main", [f"{s}/{x}-00000-of-00001.parquet" for s in ("ARC-Easy", "ARC-Challenge")
                                        for x in ("train", "validation", "test")]),
    "hellaswag": ("Rowan/hellaswag", "main", ["data/train-00000-of-00001.parquet",       # few-shot examples
                                              "data/validation-00000-of-00001.parquet"]),
    "humaneval": ("openai/openai_humaneval", "main", ["openai_humaneval/test-00000-of-00001.parquet"]),
}
# CORE eval bundle used by nanochat / pagoda (DCLM CORE tasks) -> raw/eval_bundle
CORE_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

TEACHERS = {  # name -> (repo, pinned commit); tokenizer files must hash identically (plan section 6)
    "Qwen3-0.6B": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "Qwen3-1.7B": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
}

# name -> (repo_id, revision, [files]); everything lands in raw/<name>/<file path>
HF_FILES = {
    # ProcPile (S0 formal derivations; plan section 9): 250k rows per file, many symbolic tasks with solver traces
    "procedural_pile": ("reasoning-core/procedural-pile", "0e3216537b0cb7df8709c6cae34bfceae6450068", [
        "data/train-00000.parquet",
        "data/train-00001.parquet",
    ]),
    # Reasoning Core rc1: the repo now redirects to symbolic-reasoning-env (S2 short problems with cot traces)
    "symbolic_reasoning_env": ("reasoning-core/symbolic-reasoning-env", "9314cc59dbcbd32f367716141ff660d02cd9731a", [
        "data/train-00000-of-00004.parquet",
        "data/test-00000-of-00001.parquet",
    ]),
    # FOL-Traces: 7.4M verified first-order-logic derivations; "only a small slice" (S0)
    "fol_traces": ("fol-traces/fol-traces", "e0ae2fbfa163a29f8545db75b748555c2bc276e8", [
        "basic/random_selection/segment_0.jsonl",
        "basic/random_selection/segment_1.jsonl",
        "train/curated/segment_0.jsonl",
    ]),
    # FLD.v2 natural-language deductions (S2, distillation prompts, eval); default + star configs
    "fld_v2": ("hitachi-nlp/FLD.v2", "7ead9a0429604efb473e96d4dc279bc3453ce995", [
        "data/train-00000-of-00001-81d338afabf39c32.parquet",
        "data/validation-00000-of-00001-bcc05849f05f130a.parquet",
        "data/test-00000-of-00001-9c326759ac5a4535.parquet",
        "star/train-00000-of-00001-acdbd023414e99d2.parquet",
        "star/validation-00000-of-00001-9c839bebf45d742b.parquet",
        "star/test-00000-of-00001-ce2b11c666ecc19c.parquet",
    ]),
    # SmolTalk2 reasoning (Qwen3-32B *_think) subsets for S2's thinking-mode chat (plan section 10, smoltalk2_think
    # pool). Of the SFT files named *_think only the everyday-chat-like English ones (217 MB); left out:
    # OpenThoughts3 and s1k (math/code, 26.5 GB), multilingual8 and aya (non-English; aya is ~1% English),
    # LongAlign (long context), smolagents (agent traces), systemchats (every row has a system prompt).
    # Separate dir: raw/smoltalk2 is a symlink into pagoda and must not be written to.
    "smoltalk2_think": ("HuggingFaceTB/smoltalk2", "fc6cc2103c066455aade5d7fbb346039ae36ca5e", [
        "SFT/smoltalk_everyday_convs_reasoning_Qwen3_32B_think-00000-of-00001.parquet",
        "SFT/multi_turn_reasoning_if_think-00000-of-00001.parquet",
        "SFT/table_gpt_Qwen3_32B_think-00000-of-00001.parquet",
    ]),
}

# Nemotron-CC synthetic subsets (S1, 15%). nvidia/Nemotron-CC-v2 on the Hub is gated with manual approval (403 for
# this account on 2026-10-25), so we use Nemotron-CC (v1) from Common Crawl's public bucket: the same four
# rephrasing kinds of the high-quality partition. Each part file is streamed and cut after `NEMOTRON_CHARS_PER_PART`
# characters, from parts spread over recent crawls.
NEMOTRON_BASE = "https://data.commoncrawl.org/"
NEMOTRON_PATHS = "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"
NEMOTRON_KINDS = ["diverse_qa_pairs", "knowledge_list", "distill", "extract_knowledge"]
NEMOTRON_CRAWLS = ["CC-MAIN-2023-14", "CC-MAIN-2023-40", "CC-MAIN-2024-10", "CC-MAIN-2024-26"]
NEMOTRON_CHARS_PER_PART = 60_000_000   # 4 kinds x 4 parts x 60M chars ~= 210M Qwen tokens unique (S1 needs 149M, cap 2)

# git repos of the formal-language generators (read, then re-implemented in bonsai/formal/; nothing is executed)
GENERATOR_REPOS = {
    "pre-pretraining": "https://github.com/michahu/pre-pretraining",   # k-Shuffle Dyck (no license file)
    "LAD-PPT": "https://github.com/osekilab/LAD-PPT",                 # MP-STRUCT (MIT)
}
