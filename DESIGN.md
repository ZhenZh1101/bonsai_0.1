# Bonsai-0.1 implementation contract

Read with `bonsai_plan_v0.1.md` (the plan; section numbers below refer to it). The reference implementation
is Pagoda-0.1 (a nanochat-style 32k-BPE model). Bonsai reuses pagoda's
structure, code style and data, and changes model / tokenizer / template / post-training to Qwen3.

Code style: match pagoda — a module docstring that says what the file implements and which plan section,
compact code, few comments (only for non-obvious reasons), `scripts/*.py` are thin argparse CLIs over
`bonsai/*`. Python 3.12, MLX, numpy. Run everything with `.venv/bin/python -m ...` from the repo root.
Tests: `pytest` under `tests/`, fast (< 30 s each file), no network, no dependence on the big data
(use `VocabMap.for_tests`, synthetic shards in `tmp_path`, `tiny` model preset).

Machine: MacBook Pro M4 Pro, 48 GB (14 CPU cores). Disk is tight (~65 GB free): never copy pagoda data,
symlink it; keep scratch output small.

## 0. Already written (do not rewrite; extend only if your task needs it and say so)

| file | what |
| --- | --- |
| `bonsai/common.py` | paths (`RAW_DIR`, `FILTERED_DIR`, `TOK_DIR`, `TOKENIZER_DIR`, `TEACHER_DIR`, `DATA_DIR`, `CKPT_DIR`, `LOG_DIR`, `SANDBOX_DIR`, `CONFIG_DIR`, `PROMPT_DIR`, `IDENTITY_DIR`, `PAGODA_ROOT`), `log`, `write_json`, `read_json`, `iter_jsonl`, `write_jsonl`, `load_yaml` |
| `bonsai/tokenizer.py` | `QwenTokenizer` (teacher ids), `VocabMap` (s2t / t2s / byte fallback), `Tokenizer` (student ids), `get_tokenizer()`, `CONTROL_IDS`, `BOS` |
| `bonsai/chat.py` | Qwen3 template: `pieces`, `render_text`, `render` (teacher ids + mask), `render_batch`, `THINK_BUDGET`, `THINK_STOP`, `with_soft_switch`, `effective_mode`, `budget_thinking`, `parse_tool_calls`, `python_call`, `tool_trajectory`, `PYTHON_TOOL` / `TOOLS` |
| `bonsai/model.py` | `ModelConfig`, `PRESETS`, `Bonsai` (Qwen3-isomorphic), `token_losses`, `loss_fn`, `KVCache`, `make_caches`, `build_model` |
| `bonsai/data/sources.py`, `scripts/download.py` | downloads (done: tokenizer, teachers, Nemotron-CC, ProcPile, symbolic-reasoning-env, FOL-Traces slice, FLD.v2) and pagoda symlinks |
| `scripts/vocab_subset.py` | day-0 vocabulary measurement -> `tokenizer/vocab_subset.npz` |
| `bonsai/sandbox.py` | pagoda's executor (`run_python`, `extract_answer`, `parse_number`, `numbers_equal`, `single_number`, `format_number`, `is_safe_code`, `fast_exec_many`) + `run_tool_call(call)`; the inference builder owns it from here (tests, polish) |
| `bonsai/data/decontam.py` | pagoda's 13-gram decontamination + FLD test (`build_index()`, `is_contaminated(text)`, auto-builds the index); the data builder owns it from here |

### 0.1 Id spaces and BOS

* **Teacher ids**: Qwen3, 0..151935. **Student ids**: 0..N-1, N = `Tokenizer.vocab_size` (the real subset is in
  `tokenizer/vocab_subset.npz`; N = 65,536 by the user's decision of 2026-10-25, measured out-of-subset rate 0.26%
  (`tokenizer/vocab_report.json`), so plan check 4 is relaxed to ~0.3%; still never hard-code N or uint16 in code;
  tests use `VocabMap.for_tests`). `tok.to_teacher(student)`, `tok.from_teacher(teacher,
  return_oov=True)` -> (student ids, bool mask of positions produced by an out-of-subset split). Out-of-subset
  teacher tokens are spelled losslessly with the fewest in-subset pieces (`VocabMap.split`, fallback "pieces";
  "bytes" = the plan's literal rule). Shard dtype = `tok.vmap.shard_dtype` (uint16 if N <= 65,536 else uint32),
  recorded in every manifest.
* Qwen3 has no BOS. **`<|endoftext|>` is the student's document-start token** (`tok.bos_id`): every pretraining
  document, every SFT / distillation conversation and every inference prompt starts with it (the model always
  sees it at position 0, like nanochat's BOS). Teacher scoring gets the same ids mapped with `s2t`
  (`<|endoftext|>` = 151643, natural for Qwen, which concatenates documents with it).
* Text is encoded with `tok.encode` / `qwen.encode` (control-token literals stay ordinary text). Conversations
  are encoded only via `bonsai.chat.render` / `render_batch` (teacher ids) then `tok.from_teacher`.
  Never encode a rendered template string with `encode` (it would escape the template's own control tokens).

### 0.2 Chat format (plan sections 6, 13)

Qwen3's template, see the `bonsai/chat.py` docstring. Key rules every producer of chat data follows:

* Thinking sample: assistant `reasoning_content` = thinking text (<= 128 teacher tokens; longer ones are cut
  with `budget_thinking`, which appends `THINK_STOP`). Non-thinking sample: `reasoning_content = ""` -> the
  empty block `<think>\n\n</think>\n\n`, which is supervised (the model learns to emit it), unless
  `think_prefilled=True` (the `enable_thinking=False` case: the empty block is part of the prompt).
* Soft switches: `with_soft_switch(user_text, "think" | "no_think")` appends ` /think` or ` /no_think`; the
  response follows the last switch. Without a switch the default is thinking (Qwen3 default).
* Tool use (math): system turn = the Qwen3 tools block for `TOOLS` (the single `python` tool), rendered by
  passing `tools=TOOLS`; assistant calls via `tool_calls=[python_call(code)]`; the result comes back as a
  `{"role": "tool", "content": output}` message (rendered as a user turn with `<tool_response>`); then the final
  assistant turn `Answer: <number>` (plan section 13; answers are parsed after `Answer:`). Every assistant turn
  after the last user query keeps its think block (empty in non-thinking mode) — `tool_trajectory` builds this.
* Loss masks: `render(..., train_on="last")` supervises the response (all assistant turns after the last real
  user query, incl. tool-calling steps, never tool output); `train_on="all"` every assistant turn.
* History turns before the last user query lose their thinking (template rule, plan section 6).

## 1. Files and formats

### 1.1 Pools: `raw/filtered/<pool>.jsonl` (one JSON record per line)

Reused from pagoda (symlinks, already decontaminated, schemas as in pagoda): `fineweb_edu`, `fineweb_edu_hq`,
`dclm`, `cosmopedia`, `cosmopedia_textbook`, `finemath`, `stack_python` (`{"text"}`); `tinygsm`
(`question, script, output, answer`), `gsm8k_train` (`question, cot, answer, script, output, ...`),
`omi1_code`, `omi2_cot` (`question, cot, answer`), `mc` (`question, choices, answer:int, source`),
`smoltalk_chat` (`messages` user/assistant, `source`), `sft_chat` (SmolTalk2 + smol-constraints, `messages`,
`subset`; reserved by prompt hash, disjoint from `smoltalk_chat`). See Pagoda's data preparation module
for exact fields.

New pools (producer in brackets):

| pool | schema | producer |
| --- | --- | --- |
| `nemotron_cc` | `{"text", "kind"}` | data (prepare) |
| `smoltalk2_think` | `{"messages": [user, {"role":"assistant","content","reasoning_content"}], "subset"}`, reasoning already cut to 128 teacher tokens with `budget_thinking` | data (prepare; downloads SmolTalk2 `*_think` subsets it needs) |
| `s0_shuffle_dyck`, `s0_dyck`, `s0_mp_struct` | `{"text"}` (ASCII, every symbol one Qwen token) | formal |
| `s0_derivations` | `{"text", "target_start": int}`: `premises: ... ; derived: ... ; goal: ... ; next: ...`, loss only from char `target_start` (just after `next:`) to the end | formal |
| `synth_reasoning` | `{"text", "kind"}` (arith scratchpad, iGSM-style dependency problems, FLD natural-language derivations) | formal |
| `fld_chat` | `{"user", "reasoning", "answer", "label", "depth", "config"}` (user/answer from `bonsai.formal.fld`) | formal |
| `rc_short` | `{"user", "reasoning", "answer", "task", "source"}` (reasoning <= 128 tokens) | formal |
| `identity_chat` | `{"messages", "category", "id"}` (plain user/assistant, no think fields) | identity |
| `identity_text` | `{"text", "genre"}` | identity |

Every new pool is 13-gram decontaminated against the eval sets (GSM8K test, MMLU test, ARC test, HellaSwag
val, HumanEval, FLD test) with `bonsai/data/decontam.py`, and the producer writes its counts into its own
`raw/filtered/stats_<owner>.json` (owner = data / formal / identity; never a shared file: builders run in
parallel). `tok/pool_cursors.json` is written only by the data builder's stage builds (smoke builds use
`tok/smoke/pool_cursors.json`).

### 1.2 Pretraining shards: `tok/<dir>/`

```
shard_00000.bin    student ids (manifest "dtype": uint16 or uint32); every doc starts with BOS
shard_00000.idx    uint64 doc start offsets + final sentinel (len = ndocs + 1)
shard_00000.mask   optional uint8, 1 = this token is a training target (predicted from the previous token)
manifest.json      {"bos_id", "vocab_size", "vocab_sha", "dtype": "uint16"|"uint32", "seq_len", "has_mask",
                    "total_tokens", "total_docs", "shards": [{"name", "tokens", "docs"}], "config", "sources"}
```

`vocab_sha` = `tok.vmap.sha()`;
the trainer refuses shards whose `vocab_sha` differs from the loaded tokenizer. Docs longer than `seq_len + 1`
are split at build time into BOS-prefixed chunks. Shard dirs: `tok/s0` (seq_len 512, has_mask), `tok/val_s0`,
`tok/s1`, `tok/val` (1% of every S1 source, fixed for the project), `tok/s2`, and `tok/smoke/{s0,s1,s2,val,
val_s0,offpolicy}` for smoke tests. Pool cursors (`tok/pool_cursors.json`) as in pagoda: S1 -> S2 -> prompt
pool consume shared pools without overlap.

SFT-style sets (`tok/offpolicy`, `tok/identity_sft`, `tok/smoke/offpolicy`): pagoda's format
(`train_00000.{bin,idx,mask}`, `val_*`, `train_manifest.json`, `val_manifest.json`), student ids, BOS first,
mask = `render` mask (BOS 0).

### 1.3 Configs

`configs/s0.yaml`, `s1.yaml`, `s2.yaml` (stage mixes, pagoda's schema: `shard_dir`, `total_tokens`, `seed`,
`val_fraction`/`val_dir`, `seq_len`, `sources: {name: {pool, format, weight, max_epochs, ...}}`),
`configs/offpolicy.yaml` (prompt pool + teacher generation), `configs/sft.yaml` (off-policy SFT set build),
`configs/onpolicy.yaml`, `configs/smoke/*.yaml` (tiny budgets).

S1 (plan section 10, 5-day 0.99B; build `total_tokens: 1.2e9` so a faster machine can train more, smaller
runs read a prefix): cosmopedia 0.25 (cap 2), nemotron_cc 0.15 (cap 2), synth_reasoning 0.04 (cap 2),
fineweb_edu 0.25, dclm 0.07, finemath 0.09 (cap 2), tinygsm 0.03 (`tinygsm_text`), stack_python 0.08
(`code`, max 8192 tokens), smoltalk_chat 0.04 (`chat`, non-thinking, empty think block, no switches).

S2 (5-day 353M; build `4.2e8`): fineweb_edu_hq 0.148 + identity_text 0.002, cosmopedia_textbook 0.15,
tinygsm 0.20 (`tool_chat`, non-thinking), omi2_cot 0.096 + gsm8k_train 0.004 (`cot_think_chat`: CoT as
thinking <= 128 tokens + `Answer: x`), fld_chat 0.08, rc_short 0.05 (`think_chat`), smoltalk 0.12 = smoltalk_chat
0.06 non-thinking + smoltalk2_think 0.0597 thinking + identity_chat 0.0003 (~1% of the conversations, plan 18),
mc 0.08 (`mc_chat`, non-thinking, `Answer: B`), stack_python 0.07. Every ChatML sample in S2 carries a think
block (empty or <= 128 tokens); 30% (stable hash) get the soft switch matching their mode.

S0 (plan section 9, 5-day 70M; build `8.0e7`, `seq_len: 512`): s0_shuffle_dyck 0.30, s0_dyck 0.10,
s0_mp_struct 0.20 (full loss), s0_derivations 0.40 (loss after `next:` only). ASCII only.

## 2. Module ownership (each builder owns its files; do not edit files you do not own — if you need a change
in someone else's file, write it down in your final report)

### core (training) — `bonsai/optim.py`, `bonsai/checkpoint.py`, `bonsai/dataloader.py`, `bonsai/train.py`,
`bonsai/load.py`, `scripts/train.py`, `scripts/bench.py`, `scripts/ckpt_check.py`, `scripts/s0_ablation.py`,
`scripts/export_hf.py`, `tests/test_model.py`, `tests/test_optim.py`, `tests/test_dataloader.py`,
`tests/test_checkpoint.py`, `tests/test_train.py`

* Optimizer (plan section 8): default Muon (2-D matrices inside `layers.*`) + AdamW (embedding, all RMSNorm
  weights incl. q/k norm); Muon LR `matrix_lr * (768/d)^0.5`; `--optimizer adamw` = Qwen-style AdamW on all
  params, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 on 2-D weights only, peak LR 1.5e-3 at d576 / 1.2e-3 at
  d640 (interpolate / scale otherwise), schedule `wsd` (default) or `cosine` (to 10% of peak).
* One run = S0 -> S1 -> S2 on one LR curve (plan section 8): `--tokens` total, `--s0-frac 0.05`
  (0 disables S0), `--s2-frac 0.25`; warmup 300 steps from step 0 (S0 is mostly at peak), constant through
  S1, linear decay to 0 over S2; 200-step 50/50 S1/S2 transition right before the decay starts; S0 -> S1
  is a hard switch. 2^17 tokens per step; S0 rows are 512 tokens, so S0 micro-batches have 2x rows (same
  tokens per micro-step). Muon momentum warmup 0.85 -> 0.95 over 300 steps.
* Loader: pagoda's BOS-aligned best-fit packing + doc shuffle + exact resumable state, extended with optional
  masks (targets where mask == 0 become -1). A cropped remainder of a doc whose mask is not all ones becomes
  context only (all its targets -1).
* Health (plan section 7): grad norm > 3 (`--spike-norm 3`) on 3 consecutive post-warmup steps (as a jump over
  the recent median, as pagoda) or a NaN -> roll back to the last full checkpoint, LR x0.7, skip 200 steps
  (at most 2 automatic recoveries), exactly pagoda's semantics (`--lr-scale` absolute, `--skip-steps` once).
* Checkpoints as pagoda: full every 30 min (keep 3) + step 100, bf16 milestones every 10%, Ctrl-C saves.
  Run names / dirs: `ckpt/bonsai-0.1/`, `logs/bonsai-0.1/` (5-day), `bonsai-0.1-mini` (3-day),
  `bonsai-0.1-probe-{s0,nos0}` (ablation). At the end of pretraining write `ckpt/<run>/base.safetensors`
  (bf16, meta incl. model_config) = Bonsai-0.1-base.
* Monitoring as pagoda (loss / gnorm / tok/s / peak memory / phase; val loss + bpb on `tok/val`, S0 val loss on
  `tok/val_s0`; fixed-prompt samples; CORE every 25% if `bonsai.core_eval` exists — import lazily, skip when
  missing). At the S0 -> S1 boundary log `bonsai.formal.dyck.dyck_accuracy(model, tok)` (lazy import; plan
  14: > 95% expected).
* `bonsai/load.py`: `load_model(run, source)` -> `(model, tok, meta)`; source in `base` (ckpt/<run>/base.safetensors
  or latest full ckpt), `latest`, `sft` (`sft_final.safetensors`), `onpolicy` (`onpolicy_final.safetensors`),
  `final` (`final.safetensors`, the release), or a path. Weights files carry `meta.model_config`.
* `scripts/bench.py` (plan section 4): real S1 shards (fall back to `tok/smoke/s1`), configs
  `bonsai-0.1-mini, 14x576, 12x640, bonsai-0.1, 16x640`, micro-batch 8/16/32, measured tok/s and peak memory
  after warm-up; applies the rule "tokens / non-embedding params >= 18 for 50 h (3-day) / 96 h (5-day) at
  0.85 x measured throughput" and writes `logs/bench.json` with the recommended config and `--tokens`.
* `scripts/s0_ablation.py` (plan section 9): `bonsai-0.1-probe`, 100M tokens total, with S0 (5%) vs without,
  same total tokens and same val set; reports val loss difference (>= 0.02 nat to keep S0) into
  `logs/s0_ablation.json`. It must support `--tokens` small for a smoke run.
* `scripts/export_hf.py`: Hugging Face Qwen3 layout (config.json with `model_type: qwen3`,
  `_name_or_path: bonsai-0.1`, 151,936-row embedding where rows outside S are 0 and an untied lm_head whose
  rows outside S are set so their logits are very negative: document the method), plus tokenizer files and a
  model card (plan section 18). Optional but nice.

### data (corpus build) — `bonsai/data/prepare.py`, `bonsai/data/render.py`, `bonsai/data/build.py`,
`bonsai/data/decontam.py`, `bonsai/data/convert.py`, `scripts/prepare_data.py`, `scripts/build_shards.py`,
`scripts/verify_data.py`, `configs/s0.yaml`, `configs/s1.yaml`, `configs/s2.yaml`, `configs/smoke/s*.yaml`,
`tests/test_render.py`, `tests/test_build.py`, `tests/test_tokenizer.py`, `tests/test_chat.py`
(may also extend `bonsai/data/sources.py` / `scripts/download.py` for SmolTalk2 think subsets)

* Port pagoda's `convert.py` (TinyGSM / GSM8K / OMI-1 -> runnable script + output; sandbox checks via
  `bonsai.sandbox`). `decontam.py` is already ported (section 0).
* `prepare.py`: new pools `nemotron_cc` (dedup by exact text hash, length >= 50 tokens-ish, English / ASCII
  ratio check, decontam) and `smoltalk2_think`; leave pagoda's pools alone.
* `render.py`: record -> (student ids with BOS, mask or None) for formats `text`, `code`, `tinygsm_text`,
  `chat` (with a `mode` option: `no_think` / `think` / `mixed`), `tool_chat`, `cot_think_chat`, `think_chat`,
  `mc_chat`, `s0_text` (full loss), `s0_derivation` (mask from `target_start`). Uses `chat.render_batch` for
  speed; 30% soft switches by stable hash in S2.
* `build.py`: pagoda's `build_stage` / `take` / `ShardWriter` / `write_mixed` with masks, `vocab_sha`,
  `seq_len` per stage; S1 also writes `tok/val`; S0 writes `tok/val_s0`.
* `verify_data.py`: the six acceptance checks of plan section 6 (hashes; 10k docs per source teacher encode ==
  pipeline output mapped back; assistant spans identical under the template; out-of-subset rate < 0.01% with the
  list of byte-split tokens; no control tokens from plain text + literal escaping; `enable_thinking=False`
  renders `<think>\n\n</think>\n\n` first) + pagoda's shard checks (mix deviation < 5%, 20 random decoded docs
  per stage into `logs/data_samples/`, val not overlapping train, mask sanity); writes `logs/data_report.json`.
* `tests/test_chat.py`: renderer == the Jinja template from `tokenizer/tokenizer_config.json` (strict mode) on a
  dozen conversations incl. tools, multi-turn, thinking, soft switches; ids == `qwen.encode_qwen(text)`;
  escaping. `tests/test_tokenizer.py`: VocabMap round trips, byte fallback lossless, `from_teacher` oov flags.

### formal (S0 + synthetic reasoning generators) — `bonsai/formal/__init__.py`, `bonsai/formal/dyck.py`,
`bonsai/formal/mp_struct.py`, `bonsai/formal/derivations.py`, `bonsai/formal/arith.py`, `bonsai/formal/igsm.py`,
`bonsai/formal/fld.py`, `bonsai/formal/logic.py`, `scripts/gen_formal.py`, `tests/test_formal.py`

* Read `external/pre-pretraining/src/grammar.py` (k-Shuffle Dyck; no license: re-implement, do not copy) and
  `external/LAD-PPT/src/mp_struct*.py` (MIT: port with attribution). Never execute code under `external/`.
* ASCII symbols that are each exactly one Qwen3 token (verify with `QwenTokenizer`), e.g. shuffle Dyck with k
  bracket types as ` a`..` z` / ` A`..` Z` or similar; document k and the mapping.
* `dyck.dyck_accuracy(model, tok, n=256, seed=0)` -> dict: held-out k-Dyck closing-bracket accuracy (exact) and
  k-Shuffle-Dyck valid-next-token rate at greedy argmax; must work on any `Bonsai` model (used by the trainer at
  the S0 -> S1 boundary; plan 14 target > 95%).
* `derivations.py`: ProcPile (`raw/procedural_pile`, pick the formal-logic / rewriting / lambda / FOL tasks that
  have step traces in `metadata.cot`), FOL-Traces (`raw/fol_traces`, map Unicode logic symbols to ASCII:
  `~ & | -> <-> forall exists`), optionally `reasoning_core` generators if installed -> one doc per
  (trace, step k): `premises: ... ; derived: s1 ; ... ; s(k-1) ; goal: ... ; next: sk`, <= 512 tokens.
* `fld.py`: FLD.v2 (`raw/fld_v2`, default + star configs) -> `fld_prompt(rec)` (user text), `fld_reasoning(rec)`
  (proof steps in English, <= 128 tokens or dropped), `fld_answer(rec)` (`Answer: PROVED|DISPROVED|UNKNOWN`),
  `load_fld_test(depths=(2,3,4))` for the eval. `logic.py`: synthetic natural-language 2-4 hop deductions in the
  same format (to top up S2 to its 8%).
* `arith.py` (multi-step arithmetic scratchpads), `igsm.py` (iGSM-style dependency word problems with step-by-
  step solutions): plain text docs for `synth_reasoning`.
* `scripts/gen_formal.py`: writes all formal pools (sizes from configs/s0.yaml / s1.yaml / s2.yaml targets x 1.3
  headroom; small `--smoke` sizes), plus `stats.json` keys.

### inference (sandbox, engine, evals, chat) — `bonsai/sandbox.py`, `bonsai/engine.py`, `bonsai/core_eval.py`,
`bonsai/tasks/__init__.py`, `bonsai/tasks/evals.py`, `scripts/chat_eval.py`, `scripts/chat.py`,
`scripts/manual_test.py`, `prompts/fixed_prompts.json`, `prompts/manual_tests.json`, `tests/test_sandbox.py`,
`tests/test_engine.py`, `tests/test_evals.py`

* `sandbox.py`: already ported (see section 0); keep its API stable (others import it). Add the plan checklist
  tests: 5 cases (normal, timeout, exception, no output, second call within one conversation via the engine).
* `engine.py` (works for any `Bonsai` model and the student tokenizer):
  - `generate(model, tok, prompt_ids, max_new_tokens, temperature=0, top_p=1, top_k=0, presence_penalty=0,
    stop_ids=None, seed=None) -> list[int]` (single sequence, KV cache).
  - `generate_batch(model, prompts, max_new_tokens, sampling, stop_ids, think_budget=None, ...) ->
    list[list[int]]`: left-padded batched decoding with a padding-aware SDPA mask, per-row stops; with
    `think_budget`, a row also stops when it has produced that many tokens inside an open `<think>` block.
  - `chat(model, tok, messages, enable_thinking=True, tools=None, max_tool_calls=2, think_budget=128,
    max_answer_tokens=300, sampling=None) -> ChatResult` and `chat_batch(model, tok, items, ...)` (rounds:
    generate until each row stops; rows that hit the think budget get `THINK_STOP` + `\n</think>\n\n` forced
    and continue; rows whose turn ended with a python tool call get the sandbox output appended as the exact
    template continuation (`<|im_end|>\n<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>\n
    <|im_start|>assistant\n`) and continue, at most `max_tool_calls`; re-prefill per round is fine).
    `ChatResult`: `ids` (full student ids incl. BOS + prompt + everything generated / appended), `prompt_len`,
    `sampled` (bool per position: token chosen by the model), `forced` (budget stop text, tool continuations),
    `tool_output` (bool: inside `<tool_response>` content), `text` (assistant text of the last turn),
    `messages` (conversation incl. generated turns, with reasoning_content / tool_calls), `n_tool_calls`,
    `think_tokens`, `hit_budget`, `finished` (ended with `<|im_end|>`).
  - prompt = BOS + `chat.render(..., add_generation_prompt=True, enable_thinking=...)` mapped to student ids.
* `tasks/evals.py` (plan section 15): HellaSwag (acc_norm, completion scoring on the base model, CORE-style),
  ARC-Easy/Challenge and MMLU (multiple choice in chat format, non-thinking; report accuracy and the
  always-same-letter baseline as pagoda does), GSM8K non-thinking + tool (greedy, `Answer:` parsing, numeric
  compare, at most 2 calls), GSM8K thinking <= 128 no tool, FLD test 2-4 hops (thinking mode; uses
  `bonsai.formal.fld`), mode switching (100 prompts, `/think`, `/no_think`, `enable_thinking=False` one third
  each; pass = think block non-empty / empty / prefilled as required, and a well-formed answer). Each returns a
  dict with accuracy, n, and a few samples. `core_eval.py`: port pagoda's CORE (22 ICL tasks from
  `raw/eval_bundle`) for the base model.
* `scripts/chat_eval.py --run R --source base|sft|onpolicy|final --tasks ...` -> `logs/<run>/eval_<source>.json`
  (+ markdown table); `scripts/chat.py` (interactive / single prompt, `--think/--no-think`, tools on for math);
  `scripts/manual_test.py` (plan 15: 20 pagoda manual prompts + 5 mode-switch + 5 multi-turn tool -> markdown
  for a human to grade). Prompts: copy pagoda's `prompts/*.json` and extend.

### distill (teachers, prompt pool, off-policy generation, SFT, on-policy) — `bonsai/teacher.py`,
`bonsai/sft.py`, `bonsai/distill.py`, `scripts/build_prompts.py`, `scripts/gen_offpolicy.py`,
`scripts/build_sft.py`, `scripts/sft.py`, `scripts/distill_onpolicy.py`, `scripts/identity_sft.py`,
`scripts/teacher_kl.py`, `configs/offpolicy.yaml`, `configs/sft.yaml`, `configs/onpolicy.yaml`,
`configs/smoke/{offpolicy,sft,onpolicy}.yaml`, `tests/test_distill.py`, `tests/test_teacher.py`

* `teacher.py`: `Teacher(name)` loads `teachers/<name>` with mlx-lm (bf16); `generate_batch(prompts_teacher_ids,
  sampling, max_tokens, stop)` (thinking: T 0.6 / top-p 0.95 / top-k 20; non-thinking: T 0.7 / top-p 0.8 /
  top-k 20; presence penalty 1.5 implemented as a logits processor if mlx-lm lacks it); `subset_logprobs(
  teacher_ids (B,T), vmap) -> (B,T,N)` fp32 log-softmax over S (`logits[..., s2t]` then logsumexp, plan 12),
  chunked over the batch to bound memory.
* `build_prompts.py` (plan 11): ~100k prompts: SmolTalk user turns 55% (from `sft_chat`, never seen in S1/S2),
  GSM8K train + TinyGSM (after the S2 cursor) + OpenMathInstruct-2 30%, ARC + explanation QA 10%, tool demos 5%;
  13-gram decontam; fields `{"id", "source", "kind", "messages", "answer", "tools"}`; writes
  `data/prompts.jsonl`, `data/prompts_onpolicy.jsonl` (resampled, identity-like prompts removed via
  `bonsai.identity.is_identity_prompt`), `data/prompts_heldout.jsonl` (teacher-KL eval).
* `gen_offpolicy.py` (plan 11 + 14 command): `--teacher Qwen3-1.7B --prompts data/prompts.jsonl --modes
  think,no_think --budget 128 [--shard i/n] [--limit n]`; English system prompt for the teacher only (not stored);
  thinking budget with `THINK_STOP`; math prompts with tools: teacher writes the call, `bonsai.sandbox` runs it,
  result appended as the tool turn, teacher continues (one multi-turn sample); filters: verifiable answers
  correct, non-English / mixed language, repetition, answer > 400 tokens, missing / malformed think block,
  think-answer inconsistency, Qwen / Alibaba / Tongyi self-claims (`bonsai.identity.qwen_self_claims`),
  13-gram decontam of outputs. Resumable; output `data/offpolicy/*.jsonl` + `data/offpolicy/stats.json`.
* `build_sft.py`: off-policy records + 2% identity conversations (scripted, not from the teacher; both modes)
  -> `tok/offpolicy` (SFT format, 2% val). Also `tok/identity_sft` (identity convs + general replay) for the
  post-on-policy identity fix.
* `sft.py` + `scripts/sft.py --data tok/offpolicy --epochs 2` (plan 11): from `base`, assistant-only loss,
  packing to 1024, LR 0.15 x the pretraining peak (configurable 0.1-0.2), warmup 50, linear decay, 2 epochs,
  keep epoch 1 if val loss rises; writes `ckpt/<run>/sft_final.safetensors` (Bonsai-0.1-sft). Reusable for the
  identity mini-SFT (`scripts/identity_sft.py`, 100-200 steps -> `final.safetensors`).
* `distill.py` + `scripts/distill_onpolicy.py --teacher Qwen3-0.6B --steps 150 --batch 256` (plan 12): each step
  sample prompts (mode random per prompt; tools for math), student `engine.chat_batch` (budget 128, answer <=
  300), map to teacher ids, teacher prefill, `subset_logprobs`, exact reverse KL over S at mask positions
  (`sampled & ~forced & ~tool_output & ~byte-split`), mean over masked tokens, AdamW at 0.5 x the off-policy
  LR, clip 1.0, one update per step; every 25 steps: GSM8K subset (non-thinking + tool), 20-conversation format
  check, mean KL, checkpoint; stop rules of plan 12; writes `onpolicy_final.safetensors`.
* `teacher_kl.py`: mean per-token KL(student || teacher) on held-out prompts (plan 15, three stages).
* `tests/test_distill.py`: KL against a naive implementation, mask covers byte-split tokens and tool output,
  s2t gather correct, 3-step tiny run with a fake teacher reduces KL.

### identity — `identity/` (all files), `bonsai/identity.py`, `scripts/gen_identity.py`,
`scripts/identity_eval.py`, `tests/test_identity.py`

Bonsai-0.1 replaces Pagoda-0.1 (plan 18). Reuse pagoda's identity data (persona,
1,990 conversations in 8 categories, pretrain bank + 7,312 texts, eval set, checks) and change it to Bonsai:
names, and every fact that differs (see the persona facts in the identity task brief). Must say it is not Qwen
(and not ChatGPT etc.). Exports for others: `is_identity_prompt(text)`, `qwen_self_claims(text)`,
`other_model_claims`, `says_name`, `name_count`, `system_prompt`, `with_system_prompt`, `short_reasoning(user)`
(a brief thinking text for identity samples in thinking mode), `eval_set()`. `scripts/gen_identity.py pools`
writes `raw/filtered/identity_chat.jsonl` and `identity_text.jsonl`.

## 2b. Additions recorded after the build

* Pools: `s0_derivations` records carry `source`; `rc_short` carries `cut` (reasoning cut to 128 tokens with
  THINK_STOP); `synth_reasoning` kinds are `arith`, `igsm`, `fld_text` (`logic_text` available).
* Engine extras: `ChatResult.oov / tool_calls / stop_reason / completion_ids / reasoning`; `Sampling` presets
  `GREEDY`, `THINK_SAMPLING`, `NO_THINK_SAMPLING`, `sampling="qwen"` (preset per row's mode); `generate*` keep the
  stop id; `ScriptedModel` / `scripted_policy` test doubles; `cast_for_inference` (eval-only processes);
  `span_logprobs` / `final_logits`.
* Distillation outputs: `ckpt/<run>/onpolicy/{step_XXXXXX, best.safetensors}`, `logs/<run>/onpolicy.{log,jsonl}`,
  `onpolicy_samples.jsonl`, `onpolicy_summary.json`, `teacher_kl_<source>.json` + merged `teacher_kl.json`.
  `distill_onpolicy --resume auto` recomputes the LR from the current flags (plan 16 "halve the LR":
  `--resume auto --lr-frac 0.25`). Exit code 3 = signal, checkpoint saved.
* Vocabulary subset v2 (`scripts/vocab_subset.py`): sample = the rendered training mix (S0 5%, S1 70%, S2 25%),
  every token of the template's constants forced into S (164 ids, e.g. `assistant`, `"}}\n`), vocab_sha
  `38ac1e986e5e0928`, out-of-subset 0.242% on the sample.

### After the code review (35 findings, all fixed)

* On-policy scoring and teacher KL use the teacher's generation-time context: the teacher-only system prompt of
  `configs/offpolicy.yaml` for the prompt's kind (math_tool when tools are offered), rendered by
  `bonsai.teacher.build_job`; student position i <-> teacher position i + (teacher prompt len - student prompt
  len), completion ids identical. `--set system_prompts=null` scores without it (not recommended).
* Monitors: GSM8K-style health check on the numeric math prompts of `data/prompts_heldout.jsonl`
  (`evals.eval_gsm8k_tool(..., items=...)`), format check on `evals.eval_mode_switch(..., split="dev")`
  (`prompts/fixed_prompts.json["mode_switch_dev"]`, disjoint from the reported `mode_switch_eval`); mode compliance is
  scored apart from length (cap 400, `truncated`, `strict_accuracy`). Test sets are only used for reported evals.
* Thinking budget: `chat.think_stop(tools)` (THINK_STOP, or THINK_STOP_TOOL when the python tool is offered) for
  teacher and student alike. The python tool has notebook semantics (a final bare expression's value is printed;
  `sandbox.run_tool_call`); `run_tool_call` never raises.
* S2 shards carry masks (`has_mask`): tool_chat docs have mask 0 on the tools system block (context only), all
  other docs all ones. identity_chat 0.0006 of S2 (the whole shuffled pool once), smoltalk_chat 0.0597.
* Resume everywhere: pretraining (`--resume auto`; checkpoints newer than the resume point move to
  `ckpt/<run>/abandoned_<time>/`; rollback goes to the checkpoint this process last loaded / saved; no scheduled
  saves inside a spike streak; `--skip-steps` applied once per (N, --lr-scale) request), SFT and identity SFT
  (`--resume auto`, `ckpt/<run>/{sft,identity_sft}/`, exit 3 on SIGINT/SIGTERM/SIGHUP), on-policy (`--resume
  auto|latest|best|<dir>`, a health stop or a changed --lr-frac rewinds to the best healthy checkpoint, `--overwrite`
  moves an earlier attempt aside), teacher generation (resumes by sample id, repairs a truncated last line).
* Trainer refuses a budget whose phases would repeat their shards (`--allow-repeat` to override); bench.json
  `summary[50h|96h].recommended = {config, micro_batch, tokens, tokens_flag, command, decided_by, note}`, read by
  `runs/train_5d.sh`. S0 ablation: warmup / Muon warmup / transition scaled to the probe's length (21 / 21 / 14).
* HF export fits the out-of-subset head rows on held-out text + chat positions and fails (writes nothing) above
  `--max-oov-mass` (1e-4); `generation_config.json` suppresses out-of-subset ids.

### Decisions after day 0 (2026-10-25 / 26)

* S0 ablation (scripts/s0_ablation.py, bonsai-0.1-probe, 100M tokens): val loss 3.8585 with S0 vs 3.8079 without
  (-0.051 nat, threshold +0.02) -> S0 dropped; runs/train_5d.sh adds `--s0-frac 0` from logs/s0_ablation.json.
* 5-day budget: bonsai-0.1 (14 x 640) at **2.0B tokens** (the plan said 1.41B): 15,259 steps, S1 1.487B -> S2 513M;
  shards rebuilt with configs/s1.yaml 1.52e9 and s2.yaml 5.25e8 (identity_chat 0.0004, smoltalk_chat 0.0599 with
  max_epochs 1.5). Identity facts: about 2 billion training tokens, no formal-language warm-up (tested on a small
  model and dropped).

## 3. Integration (done by the lead after the builders)

`runs/*.sh`, `README.md`, `scripts/smoke_test.py`, running the real data pipeline, end-to-end smoke tests.
